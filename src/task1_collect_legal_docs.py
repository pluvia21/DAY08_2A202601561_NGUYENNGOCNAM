"""
Task 1 — Thu thập văn bản chính sách/quy định dịch vụ đại học.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản chính sách (PDF/DOCX) từ trang công khai của một trường đại học.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, mô tả đúng nội dung.

Gợi ý nguồn (trang công khai Trường Đại học Công nghệ - ĐHQGHN, uet.vnu.edu.vn):
    - Học phí & phương thức thanh toán (Tuition Fees)
    - Quy định học bổng khuyến khích học tập (Scholarship eligibility)
    - Quy định ký túc xá / hỗ trợ chỗ ở (Accommodation Services)
    - Hướng dẫn đăng ký học phần qua cổng thông tin sinh viên (Course Registration)
"""

import io
from pathlib import Path
import sys
from typing import List, Optional
import requests

# Xử lý encoding cho stdout trên Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory() -> Path:
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def download_file(url: str, filename: str, timeout: int = 15) -> Optional[Path]:
    """
    Tải file PDF/DOCX từ URL và lưu vào DATA_DIR.

    Args:
        url: Link tải trực tiếp file văn bản.
        filename: Tên file lưu lại (không dấu, rõ ràng).
        timeout: Thời gian chờ tối đa (giây).

    Returns:
        Path của file đã tải thành công, hoặc None nếu thất bại.
    """
    setup_directory()
    filepath = DATA_DIR / filename
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        filepath.write_bytes(response.content)
        print(f"✓ Đã tải thành công: {filepath.name} ({len(response.content)} bytes)")
        return filepath
    except Exception as e:
        print(f"✗ Lỗi khi tải {url}: {e}")
        return None


def verify_legal_docs() -> List[Path]:
    """
    Kiểm tra danh sách các file hợp lệ (.pdf, .docx, .doc với dung lượng > 1KB) trong DATA_DIR.

    Returns:
        Danh sách các file hợp lệ.
    """
    setup_directory()
    valid_extensions = {".pdf", ".docx", ".doc"}
    valid_files: List[Path] = []

    for f in DATA_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in valid_extensions:
            if f.stat().st_size > 1024:
                valid_files.append(f)
            else:
                print(f"⚠️ File {f.name} quá nhỏ ({f.stat().st_size} bytes), có thể bị lỗi.")

    return valid_files


def main():
    """Thu thập và xác minh các văn bản pháp luật / quy định dịch vụ đại học."""
    print("=" * 60)
    print("Task 1: Thu thập văn bản chính sách/quy định dịch vụ đại học")
    print("=" * 60)

    setup_directory()
    valid_files = verify_legal_docs()

    print(f"\n📁 Tìm thấy {len(valid_files)} file văn bản hợp lệ trong {DATA_DIR}:")
    for idx, filepath in enumerate(valid_files, 1):
        size_kb = filepath.stat().st_size / 1024
        print(f"  {idx}. {filepath.name:<40} | {size_kb:.1f} KB | {filepath.suffix}")

    if len(valid_files) >= 3:
        print(f"\n✅ Hoàn thành Task 1! Đã thu thập đủ {len(valid_files)}/3 văn bản quy định.")
    else:
        print(f"\n⚠️ Chưa đủ 3 văn bản! Hiện có: {len(valid_files)} file. Vui lòng thêm/tải thêm văn bản.")


if __name__ == "__main__":
    main()

