#!/usr/bin/env python3
"""Local batch colour-processing and review web application."""

from __future__ import annotations

import argparse
import io
import json
import mimetypes
import os
import shutil
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from PIL import Image, ImageCms

from color_model import load_color_model, render_cmyk_to_srgb


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_filename(name: str) -> str:
    name = Path(unquote(name)).name.strip().replace("\x00", "")
    stem = "".join(c for c in Path(name).stem if c.isalnum() or c in "-_. ").strip()
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("仅支持 JPG、PNG 和 TIFF 图片")
    return (stem or "image")[:120] + suffix


class AppState:
    def __init__(
        self, model_path: Path, data_dir: Path, workers: int = 2,
        max_hue_shift: float = 15.0, max_upload_mb: int = 512,
        edge_lift: float | None = None,
        shadow_lift: float | None = None,
        device: str | None = "auto",
    ):
        self.model = load_color_model(model_path, device=device)
        self.model_path = model_path
        self.data_dir = data_dir
        self.max_hue_shift = max_hue_shift
        self.max_upload_mb = max_upload_mb
        self.edge_lift = edge_lift
        self.shadow_lift = shadow_lift
        self.max_upload_bytes = max_upload_mb * 1024 * 1024
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.infer_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="colour")

    def batch_dir(self, batch_id: str) -> Path:
        if not batch_id.replace("-", "").isalnum():
            raise ValueError("非法批次 ID")
        return self.data_dir / batch_id

    def manifest_path(self, batch_id: str) -> Path:
        return self.batch_dir(batch_id) / "manifest.json"

    def read_batch(self, batch_id: str) -> dict:
        with self.lock:
            path = self.manifest_path(batch_id)
            if not path.exists():
                raise FileNotFoundError(batch_id)
            return json.loads(path.read_text("utf-8"))

    def write_batch(self, manifest: dict) -> None:
        with self.lock:
            path = self.manifest_path(manifest["id"])
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
            tmp.replace(path)

    def create_batch(self, name: str) -> dict:
        batch_id = uuid.uuid4().hex[:12]
        root = self.batch_dir(batch_id)
        for child in ("input", "output", "preview", "target", "target_preview"):
            (root / child).mkdir(parents=True, exist_ok=True)
        manifest = {
            "id": batch_id,
            "name": name.strip()[:80] or f"调色批次 {datetime.now():%m-%d %H:%M}",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "model": self.model_path.name,
            "target_profile": self.model.metadata.get("target_profile", "CMYK"),
            "max_hue_shift": self.max_hue_shift,
            "images": [],
        }
        self.write_batch(manifest)
        return manifest

    def list_batches(self) -> list[dict]:
        rows = []
        for path in self.data_dir.glob("*/manifest.json"):
            try:
                batch = json.loads(path.read_text("utf-8"))
                counts = self.counts(batch)
                rows.append({k: batch[k] for k in ("id", "name", "created_at", "updated_at")}
                            | {"counts": counts, "target_profile": batch.get("target_profile")})
            except (OSError, ValueError, KeyError):
                continue
        return sorted(rows, key=lambda x: x["created_at"], reverse=True)

    @staticmethod
    def counts(batch: dict) -> dict:
        images = batch["images"]
        return {
            "total": len(images),
            "completed": sum(x["process_status"] == "completed" for x in images),
            "processing": sum(x["process_status"] in {"queued", "processing"} for x in images),
            "failed": sum(x["process_status"] == "failed" for x in images),
            "approved": sum(x["review_status"] == "approved" for x in images),
            "rejected": sum(x["review_status"] == "rejected" for x in images),
            "pending": sum(x["review_status"] == "pending" for x in images),
        }

    def add_upload_file(
        self, batch_id: str, filename: str, uploaded_path: Path, size_bytes: int
    ) -> dict:
        filename = safe_filename(filename)
        image_id = uuid.uuid4().hex[:12]
        stored_name = image_id + Path(filename).suffix
        path = self.batch_dir(batch_id) / "input" / stored_name
        try:
            with Image.open(uploaded_path) as image:
                image.verify()
            with Image.open(uploaded_path) as image:
                width, height = image.size
        except Exception:
            raise ValueError("上传内容不是有效图片")
        uploaded_path.replace(path)

        with self.lock:
            batch = self.read_batch(batch_id)
            item = {
                "id": image_id,
                "filename": filename,
                "input_file": stored_name,
                "output_file": None,
                "preview_file": None,
                "target_file": None,
                "target_preview_file": None,
                "target_filename": None,
                "width": width,
                "height": height,
                "size_bytes": size_bytes,
                "process_status": "queued",
                "review_status": "pending",
                "note": "",
                "error": None,
                "created_at": utc_now(),
                "processed_at": None,
            }
            batch["images"].append(item)
            batch["updated_at"] = utc_now()
            self.write_batch(batch)
        self.executor.submit(self.process_image, batch_id, image_id)
        return item

    def update_item(self, batch_id: str, image_id: str, **changes) -> dict:
        with self.lock:
            batch = self.read_batch(batch_id)
            item = next((x for x in batch["images"] if x["id"] == image_id), None)
            if not item:
                raise FileNotFoundError(image_id)
            item.update(changes)
            batch["updated_at"] = utc_now()
            self.write_batch(batch)
            return item.copy()

    def process_image(self, batch_id: str, image_id: str) -> None:
        item = self.update_item(batch_id, image_id, process_status="processing", error=None)
        root = self.batch_dir(batch_id)
        try:
            with Image.open(root / "input" / item["input_file"]) as source:
                with self.infer_lock:
                    result = self.model.predict_image(
                        source,
                        max_hue_shift=self.max_hue_shift,
                        edge_lift=self.edge_lift,
                        shadow_lift=self.shadow_lift,
                    )
            output_name = image_id + ".tif"
            preview_name = image_id + ".jpg"
            result.save(root / "output" / output_name, compression="tiff_lzw", icc_profile=self.model.target_icc)
            preview = render_cmyk_to_srgb(result, self.model.target_icc)
            preview.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
            preview.save(root / "preview" / preview_name, quality=91, optimize=True)
            self.update_item(
                batch_id, image_id,
                output_file=output_name,
                preview_file=preview_name,
                process_status="completed",
                processed_at=utc_now(),
            )
        except FileNotFoundError:
            return
        except Exception as exc:
            try:
                self.update_item(batch_id, image_id, process_status="failed", error=str(exc))
            except FileNotFoundError:
                return

    def review(self, batch_id: str, image_id: str, status: str, note: str) -> dict:
        if status not in {"pending", "approved", "rejected"}:
            raise ValueError("非法审核状态")
        return self.update_item(batch_id, image_id, review_status=status, note=note[:500])

    def delete_batch(self, batch_id: str) -> dict:
        """Remove a batch directory and all of its uploaded/processed files."""
        root = self.batch_dir(batch_id)
        with self.lock:
            if not self.manifest_path(batch_id).exists():
                raise FileNotFoundError(batch_id)
            shutil.rmtree(root)
        return {"ok": True, "id": batch_id}

    def add_target_file(
        self, batch_id: str, image_id: str, filename: str, uploaded_path: Path
    ) -> dict:
        """Attach an aligned RGB/CMYK reference image to one review item."""
        filename = safe_filename(filename)
        batch = self.read_batch(batch_id)
        item = next((x for x in batch["images"] if x["id"] == image_id), None)
        if not item:
            raise FileNotFoundError(image_id)

        root = self.batch_dir(batch_id)
        target_dir = root / "target"
        preview_dir = root / "target_preview"
        target_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)
        stored_name = image_id + Path(filename).suffix
        preview_name = image_id + ".jpg"
        path = target_dir / stored_name
        try:
            with Image.open(uploaded_path) as target:
                if target.size != (item["width"], item["height"]):
                    raise ValueError(
                        f"目标图尺寸必须与原图一致：需要 {item['width']}×{item['height']}，"
                        f"实际为 {target.width}×{target.height}"
                    )
                # Generate only a review-size image before ICC conversion. A
                # full 24 MP CMYK + RGB render can otherwise exceed 200 MB.
                target.draft(target.mode, (1800, 1800))
                target.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
                if target.mode == "CMYK":
                    icc = target.info.get("icc_profile") or self.model.target_icc
                    preview = render_cmyk_to_srgb(target, icc)
                else:
                    icc = target.info.get("icc_profile")
                    if icc:
                        try:
                            src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                            preview = ImageCms.profileToProfile(
                                target.convert("RGB"), src_profile, ImageCms.createProfile("sRGB"),
                                outputMode="RGB",
                            )
                        except Exception:
                            preview = target.convert("RGB")
                    else:
                        preview = target.convert("RGB")
                preview.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
                preview.save(preview_dir / preview_name, quality=91, optimize=True)
        except Exception:
            (preview_dir / preview_name).unlink(missing_ok=True)
            raise

        old_target = item.get("target_file")
        if old_target and old_target != stored_name:
            (target_dir / old_target).unlink(missing_ok=True)
        uploaded_path.replace(path)
        return self.update_item(
            batch_id, image_id,
            target_file=stored_name,
            target_preview_file=preview_name,
            target_filename=filename,
        )

    def make_zip(self, batch_id: str, status: str) -> Path:
        batch = self.read_batch(batch_id)
        if status not in {"all", "approved", "rejected", "pending"}:
            raise ValueError("非法筛选条件")
        path = self.batch_dir(batch_id) / f"{status}_outputs.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
            for item in batch["images"]:
                if not item["output_file"] or (status != "all" and item["review_status"] != status):
                    continue
                source = self.batch_dir(batch_id) / "output" / item["output_file"]
                zf.write(source, arcname=Path(item["filename"]).stem + "_CMYK.tif")
            zf.writestr("review_manifest.json", json.dumps(batch, ensure_ascii=False, indent=2))
        return path


class Handler(BaseHTTPRequestHandler):
    server_version = "ColorReview/1.0"

    @property
    def app(self) -> AppState:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def json_response(self, data, status=200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def error_response(self, message: str, status=400) -> None:
        self.json_response({"error": message}, status)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 64 * 1024:
            raise ValueError("请求过大")
        return json.loads(self.rfile.read(length) or b"{}")

    def receive_upload(self, path: Path, length: int) -> None:
        """Stream an HTTP request body to disk without retaining it in RAM."""
        remaining = length
        with path.open("wb") as output:
            while remaining:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ConnectionError("上传连接提前中断")
                output.write(chunk)
                remaining -= len(chunk)

    def send_file(self, path: Path, download_name: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        if download_name:
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(download_name)}")
        self.end_headers()
        with path.open("rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            parts = [x for x in parsed.path.split("/") if x]
            if parsed.path == "/api/config":
                self.json_response({
                    "max_upload_mb": self.app.max_upload_mb,
                    "max_upload_bytes": self.app.max_upload_bytes,
                })
            elif parsed.path == "/api/batches":
                self.json_response(self.app.list_batches())
            elif len(parts) == 3 and parts[:2] == ["api", "batches"]:
                batch = self.app.read_batch(parts[2])
                batch["counts"] = self.app.counts(batch)
                self.json_response(batch)
            elif len(parts) == 4 and parts[:2] == ["api", "batches"] and parts[3] == "download":
                status = parse_qs(parsed.query).get("status", ["approved"])[0]
                path = self.app.make_zip(parts[2], status)
                self.send_file(path, f"{parts[2]}_{status}.zip")
            elif len(parts) == 5 and parts[0] == "media":
                batch_id, image_id, kind = parts[1], parts[2], parts[3]
                batch = self.app.read_batch(batch_id)
                item = next(x for x in batch["images"] if x["id"] == image_id)
                mapping = {
                    "input": ("input", "input_file"),
                    "preview": ("preview", "preview_file"),
                    "output": ("output", "output_file"),
                    "target": ("target", "target_file"),
                    "target-preview": ("target_preview", "target_preview_file"),
                }
                if kind not in mapping or not item.get(mapping[kind][1]):
                    raise FileNotFoundError(kind)
                folder, key = mapping[kind]
                self.send_file(self.app.batch_dir(batch_id) / folder / item[key], item["filename"] if kind == "output" else None)
            elif parsed.path.startswith("/api/"):
                self.send_error(404)
            else:
                name = "index.html" if parsed.path in {"/", "/review"} else parsed.path.lstrip("/")
                static = (Path(__file__).parent / "web" / name).resolve()
                web_root = (Path(__file__).parent / "web").resolve()
                if web_root not in static.parents and static != web_root:
                    self.send_error(403)
                else:
                    self.send_file(static)
        except FileNotFoundError:
            self.error_response("未找到资源", 404)
        except (ValueError, KeyError, StopIteration) as exc:
            self.error_response(str(exc), 400)
        except Exception as exc:
            self.error_response(f"服务器错误：{exc}", 500)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            parts = [x for x in parsed.path.split("/") if x]
            if parsed.path == "/api/batches":
                self.json_response(self.app.create_batch(self.read_json().get("name", "")), 201)
                return
            if len(parts) == 4 and parts[:2] == ["api", "batches"] and parts[3] == "images":
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > self.app.max_upload_bytes:
                    raise ValueError(f"单张图片大小必须在 {self.app.max_upload_mb} MB 以内")
                filename = parse_qs(parsed.query).get("filename", [""])[0]
                upload_dir = self.app.batch_dir(parts[2]) / "input"
                upload_dir.mkdir(parents=True, exist_ok=True)
                temporary = upload_dir / f".{uuid.uuid4().hex}.upload"
                try:
                    self.receive_upload(temporary, length)
                    item = self.app.add_upload_file(parts[2], filename, temporary, length)
                finally:
                    temporary.unlink(missing_ok=True)
                self.json_response(item, 202)
                return
            if len(parts) == 5 and parts[:2] == ["api", "batches"] and parts[4] == "target":
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > self.app.max_upload_bytes:
                    raise ValueError(f"目标图大小必须在 {self.app.max_upload_mb} MB 以内")
                filename = parse_qs(parsed.query).get("filename", [""])[0]
                upload_dir = self.app.batch_dir(parts[2]) / "target"
                upload_dir.mkdir(parents=True, exist_ok=True)
                temporary = upload_dir / f".{parts[3]}.{uuid.uuid4().hex}.upload"
                try:
                    self.receive_upload(temporary, length)
                    item = self.app.add_target_file(parts[2], parts[3], filename, temporary)
                finally:
                    temporary.unlink(missing_ok=True)
                self.json_response(item, 201)
                return
            self.send_error(404)
        except FileNotFoundError:
            self.error_response("批次不存在", 404)
        except (ValueError, json.JSONDecodeError) as exc:
            self.error_response(str(exc), 400)
        except Exception as exc:
            self.error_response(f"服务器错误：{exc}", 500)

    def do_PATCH(self) -> None:
        try:
            parts = [x for x in urlparse(self.path).path.split("/") if x]
            if len(parts) != 5 or parts[:2] != ["api", "batches"] or parts[4] != "review":
                self.send_error(404)
                return
            data = self.read_json()
            item = self.app.review(parts[2], parts[3], data.get("status", "pending"), data.get("note", ""))
            self.json_response(item)
        except FileNotFoundError:
            self.error_response("图片不存在", 404)
        except (ValueError, json.JSONDecodeError) as exc:
            self.error_response(str(exc), 400)

    def do_DELETE(self) -> None:
        try:
            parts = [x for x in urlparse(self.path).path.split("/") if x]
            if len(parts) != 3 or parts[:2] != ["api", "batches"]:
                self.send_error(404)
                return
            self.json_response(self.app.delete_batch(parts[2]))
        except FileNotFoundError:
            self.error_response("批次不存在", 404)
        except ValueError as exc:
            self.error_response(str(exc), 400)
        except Exception as exc:
            self.error_response(f"服务器错误：{exc}", 500)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="5D2A8056_model.npz")
    p.add_argument("--data", default="web_data")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument(
        "--max-hue-shift", type=float, default=15.0,
        help="polynomial model hue protection; residual-LUT models use confidence fallback",
    )
    p.add_argument(
        "--max-upload-mb", type=int, default=512,
        help="maximum size of one input or target image in MB",
    )
    p.add_argument(
        "--edge-lift", type=float, default=None,
        help="silhouette K lift 0..1; default 0.05 from the residual-LUT model. 0 disables",
    )
    p.add_argument(
        "--shadow-lift", type=float, default=None,
        help="dark-tone K lift 0..1; default 0 (off). Use e.g. 0.06 to restore",
    )
    p.add_argument(
        "--device", default="auto",
        help="PyTorch device for .pt models: auto, cpu, mps, or cuda",
    )
    args = p.parse_args()
    if args.max_upload_mb <= 0:
        raise ValueError("--max-upload-mb must be greater than zero")
    if args.edge_lift is not None and args.edge_lift < 0:
        raise ValueError("--edge-lift 不能为负数")
    if args.shadow_lift is not None and args.shadow_lift < 0:
        raise ValueError("--shadow-lift 不能为负数")
    app = AppState(
        Path(args.model), Path(args.data), args.workers,
        args.max_hue_shift, args.max_upload_mb, args.edge_lift, args.shadow_lift,
        device=args.device,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.app = app  # type: ignore[attr-defined]
    print(f"Color Review running at http://{args.host}:{args.port}")
    print(f"Model: {Path(args.model).resolve()}")
    print(f"Device: {getattr(app.model, 'device', args.device)}")
    if args.edge_lift is not None:
        print(f"Edge lift: {args.edge_lift:g}")
    if args.shadow_lift is not None:
        print(f"Shadow lift: {args.shadow_lift:g}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.executor.shutdown(wait=True)
        server.server_close()


if __name__ == "__main__":
    main()
