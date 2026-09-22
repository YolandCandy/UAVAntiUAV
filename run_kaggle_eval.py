#!/usr/bin/env python3
"""
run_kaggle_eval.py
==================
Script điều phối chạy đánh giá toàn diện mô hình UAV-Anti-UAV ReID trên Kaggle:
  1. Tự động kiểm tra và cấu hình môi trường đường dẫn Kaggle.
  2. Tiền xử lý dữ liệu (nếu chưa có query_test.json/gallery_test.json).
  3. Hiệu chuẩn ngưỡng (Threshold Calibration - Fixed FAR 0.1%).
  4. Đánh giá ReID toàn cục (Rank-1, Rank-5, mAP, mINP, TAR@FAR) trên 2 không gian (fused vs pre_bn).
  5. Chạy suy luận chuỗi video tracking thời gian thực (FSM 4 trạng thái, đo FPS, Latency, False Alarms).
  6. Kiểm tra tính bền vững chống UAV giả mạo (Robustness with Imposters).
  7. Xuất báo cáo tổng hợp trực quan.

Usage:
  python run_kaggle_eval.py --config configs/config_kaggle.yaml --mode all
  python run_kaggle_eval.py --mode calib_eval
  python run_kaggle_eval.py --mode infer --seq UAV-Anti-UAV_Test_000059
"""

import os
import sys
import glob
import shutil
import argparse
import subprocess
import yaml
import json
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Kaggle Evaluation Orchestrator for UAV-Anti-UAV ReID")
    parser.add_argument("--config", type=str, default="configs/config_kaggle.yaml", help="Path to config yaml")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "pipeline", "calib", "eval", "calib_eval", "infer", "robustness", "summary"],
                        help="Chế độ chạy")
    parser.add_argument("--seq", type=str, default=None, help="Tên sequence cụ thể để chạy infer (vd: UAV-Anti-UAV_Test_000059)")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn file trọng số best_model.pth")
    parser.add_argument("--raw-data", type=str, default=None, help="Đường dẫn dataset thô")
    return parser.parse_args()

def find_file_in_kaggle(filename, search_dirs):
    """Tìm file trong các thư mục đầu vào của Kaggle."""
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        matches = glob.glob(os.path.join(d, "**", filename), recursive=True)
        if matches:
            return matches[0]
    return None

def auto_detect_kaggle_paths(cfg, custom_weights=None, custom_raw_data=None):
    """Tự động phát hiện và liên kết các file trọng số, dữ liệu trên Kaggle."""
    print("[1/6] Kiểm tra và tự động phát hiện đường dẫn trên Kaggle...")
    
    # 1. Tìm dataset thô
    search_data_roots = [
        custom_raw_data,
        cfg.get('paths', {}).get('raw_data_dir'),
        "/kaggle/input/datasets/yolandcandy/uav-anti-uav/Test-002",
        "/kaggle/input/datasets/yolandcandy/uav-anti-uav/Test-002/Test",
        "/kaggle/input/uav-anti-uav",
        "/kaggle/input/uav-anti-uav/UAV-Anti-UAV",
        "./data/UAV-Anti-UAV"
    ]
    raw_data_path = None
    for p in search_data_roots:
        if p and os.path.exists(p):
            # Kiểm tra xem có thư mục Test bên trong không
            if os.path.exists(os.path.join(p, "Test")):
                raw_data_path = p
                break
            elif os.path.basename(p).lower() == "test":
                raw_data_path = os.path.dirname(p)
                break
            else:
                # Tìm đệ quy thư mục Test
                test_dirs = glob.glob(os.path.join(p, "**", "Test"), recursive=True)
                if test_dirs:
                    raw_data_path = os.path.dirname(test_dirs[0])
                    break
    
    if raw_data_path:
        print(f"  -> Đã tìm thấy dataset thô tại: {raw_data_path}")
        cfg['paths']['raw_data_dir'] = raw_data_path
        cfg['infer']['test_dir'] = os.path.join(raw_data_path, "Test")
        cfg['infer_robustness']['data_root'] = os.path.join(raw_data_path, "Test")
    else:
        print("  ⚠️ Không tự động tìm thấy thư mục Test của dataset UAV-Anti-UAV.")

    # 2. Tìm file trọng số (weights)
    search_weight_dirs = [
        custom_weights,
        cfg.get('eval', {}).get('model_path'),
        "/kaggle/input",
        "/kaggle/working",
        "./checkpoints",
        "."
    ]
    weights_path = None
    if custom_weights and os.path.exists(custom_weights):
        weights_path = custom_weights
    else:
        for w_cand in ["best_model.pth", "best_model_dino_convnext.pth", "best_model.pth.zip"]:
            found = find_file_in_kaggle(w_cand, search_weight_dirs)
            if found:
                weights_path = found
                break
    
    working_weight_path = "/kaggle/working/best_model.pth" if os.path.exists("/kaggle") else "./checkpoints/best_model.pth"
    os.makedirs(os.path.dirname(working_weight_path), exist_ok=True)

    if weights_path:
        print(f"  -> Đã tìm thấy trọng số tại: {weights_path}")
        # Nếu là file zip, giải nén
        if weights_path.endswith(".zip"):
            print("  -> Đang giải nén trọng số từ .zip...")
            import zipfile
            with zipfile.ZipFile(weights_path, 'r') as zf:
                zf.extractall(os.path.dirname(working_weight_path))
            # Nếu sau giải nén tạo thư mục hoặc file
            if os.path.exists(working_weight_path):
                weights_path = working_weight_path
            else:
                candidates = glob.glob(os.path.join(os.path.dirname(working_weight_path), "**", "*.pth"), recursive=True)
                if candidates:
                    weights_path = candidates[0]
        elif weights_path != working_weight_path and not os.path.exists(working_weight_path):
            try:
                shutil.copy2(weights_path, working_weight_path)
                weights_path = working_weight_path
            except Exception as e:
                print(f"  (Giữ nguyên path {weights_path}, không copy được: {e})")
        
        cfg['eval']['model_path'] = weights_path
        cfg['infer']['model_path'] = weights_path
        cfg['infer_robustness']['model_path'] = weights_path
    else:
        print(f"  ⚠️ Cảnh báo: Chưa tìm thấy file weights .pth. Vui lòng kiểm tra lại input.")

    return cfg

def run_command(cmd, desc):
    """Chạy lệnh subprocess và in tiến độ trực tiếp."""
    print(f"\n{'='*60}")
    print(f"  🚀 BẮT ĐẦU: {desc}")
    print(f"  Lệnh: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"\n❌ Lỗi khi thực hiện: {desc} (Exit code: {ret.returncode})")
    else:
        print(f"\n✅ Hoàn thành xuất sắc: {desc}")
    return ret.returncode == 0

def step_data_pipeline(cfg, config_path):
    """Chạy trích xuất crops nếu chưa có file query/gallery test."""
    query_json = cfg['eval']['query_json']
    gallery_json = cfg['eval']['gallery_json']
    if os.path.exists(query_json) and os.path.exists(gallery_json):
        print(f"-> Đã có sẵn query_test.json và gallery_test.json. Bỏ qua bước data_pipeline.")
        return True
    
    cmd = [
        sys.executable, "data_pipeline.py",
        "--config", config_path
    ]
    return run_command(cmd, "Trích xuất Dataset Query & Gallery (data_pipeline.py)")

def step_calibrate(cfg, config_path):
    """Chạy calibrate threshold Fixed-FAR 0.1%."""
    cmd = [
        sys.executable, "calibrate_threshold.py",
        "--config", config_path,
        "--cal-ratio", str(cfg.get('calibration', {}).get('cal_ratio', 0.4)),
        "--far-target", str(cfg.get('calibration', {}).get('far_target', 0.001)),
        "--output-dir", cfg.get('calibration', {}).get('output_dir', 'calib_results')
    ]
    return run_command(cmd, "Hiệu chuẩn ngưỡng Fixed-FAR 0.1% (calibrate_threshold.py)")

def step_evaluate_reid(cfg, config_path):
    """Đánh giá toàn cục ReID trên không gian fused và pre_bn."""
    calib_file = cfg.get('eval', {}).get('threshold_file')
    cmd_fused = [
        sys.executable, "evaluate_reid.py",
        "--config", config_path,
        "--space", "fused"
    ]
    if calib_file and os.path.exists(calib_file):
        cmd_fused.extend(["--threshold-file", calib_file])
    ok1 = run_command(cmd_fused, "Đánh giá Global ReID (Không gian FUSED - sau BNNeck)")

    cmd_prebn = [
        sys.executable, "evaluate_reid.py",
        "--config", config_path,
        "--space", "pre_bn"
    ]
    if calib_file and os.path.exists(calib_file):
        cmd_prebn.extend(["--threshold-file", calib_file])
    ok2 = run_command(cmd_prebn, "Đánh giá Global ReID (Không gian PRE-BN - trước BNNeck)")
    return ok1 and ok2

def step_infer_sequence(cfg, config_path, specific_seq=None):
    """Chạy mô phỏng tracking FSM thời gian thực."""
    inf_cfg = cfg.get('infer', {})
    test_dir = inf_cfg.get('test_dir') or os.path.join(cfg.get('paths', {}).get('raw_data_dir', ''), 'Test')
    
    if specific_seq:
        seq_target = os.path.join(test_dir, specific_seq) if not os.path.isabs(specific_seq) else specific_seq
    else:
        # Nếu config chỉ định 'all' hoặc một sequence cụ thể
        configured_seq = inf_cfg.get('seq_dir', 'all')
        if configured_seq.lower() == 'all':
            seq_target = 'all'
        else:
            seq_target = os.path.join(test_dir, configured_seq) if not os.path.isabs(configured_seq) else configured_seq
    
    cmd = [
        sys.executable, "infer.py",
        "--config", config_path,
        "--seq-dir", seq_target
    ]
    return run_command(cmd, f"Mô phỏng Tracking & Re-identification Inference (infer.py --seq-dir {seq_target})")

def step_robustness(cfg, config_path):
    """Chạy bài kiểm tra khả năng kháng nhiễu Imposter."""
    cmd = [
        sys.executable, "evaluate_reid_robustness.py",
        "--config", config_path
    ]
    return run_command(cmd, "Kiểm tra tính kháng nhiễu Imposter Attack (evaluate_reid_robustness.py)")

def print_final_summary(cfg):
    """In bảng tổng hợp các chỉ số đã đo đạc được."""
    print("\n" + "="*70)
    print("           BÁO CÁO TỔNG HỢP CHỈ SỐ MÔ HÌNH UAV-ANTI-UAV REID")
    print("="*70)
    
    # 1. Báo cáo Calibration
    calib_json = os.path.join(cfg.get('calibration', {}).get('output_dir', ''), "calibrated_threshold.json")
    if os.path.exists(calib_json):
        try:
            with open(calib_json, 'r') as f:
                c_data = json.load(f)
            print("\n[1] FIXED-FAR THRESHOLD CALIBRATION:")
            print(f"  - Target FAR          : {c_data.get('far_target', 0.001)*100:.2f}%")
            print(f"  - Calibrated t* (fused): {c_data.get('thresholds', {}).get('fused', c_data.get('threshold')):.4f}")
            if 'pre_bn' in c_data.get('thresholds', {}):
                print(f"  - Calibrated t* (pre_bn): {c_data.get('thresholds', {}).get('pre_bn'):.4f}")
            print(f"  - Eval Genuine Pairs  : {c_data.get('eval_genuine_pairs', 'N/A')}")
            print(f"  - Eval Imposter Pairs : {c_data.get('eval_imposter_pairs', 'N/A')}")
        except Exception as e:
            pass

    # 2. Báo cáo Global ReID Evaluation
    eval_json = os.path.join(cfg.get('eval', {}).get('output_dir', ''), "evaluation_report.json")
    if os.path.exists(eval_json):
        try:
            with open(eval_json, 'r') as f:
                e_data = json.load(f)
            print("\n[2] GLOBAL RE-IDENTIFICATION METRICS:")
            print(f"  {'Không gian':<12} | {'Rank-1 (%)':<10} | {'Rank-5 (%)':<10} | {'mAP (%)':<10} | {'mINP (%)':<10} | {'TAR@FAR 0.1%':<12}")
            print("  " + "-"*65)
            spaces = e_data.get('spaces', {})
            if spaces:
                for sp_name, m in spaces.items():
                    print(f"  {sp_name:<12} | {m.get('rank1', 0):<10.2f} | {m.get('rank5', 0):<10.2f} | {m.get('map', 0):<10.2f} | {m.get('minp', 0):<10.2f} | {m.get('tar_at_far', 0):<12.2f}")
            else:
                print(f"  {'Primary':<12} | {e_data.get('rank1', 0):<10.2f} | {e_data.get('rank5', 0):<10.2f} | {e_data.get('map', 0):<10.2f} | {e_data.get('minp', 0):<10.2f} | {e_data.get('tar_at_far', 0):<12.2f}")
        except Exception as e:
            pass

    # 3. Báo cáo Realtime Inference
    infer_summary = os.path.join(cfg.get('infer', {}).get('out_dir', ''), "summary_metrics.txt")
    if os.path.exists(infer_summary):
        print("\n[3] SEQUENTIAL TRACKING & REALTIME INFERENCE:")
        with open(infer_summary, 'r') as f:
            for line in f:
                print("  " + line.strip())

    print("\n" + "="*70 + "\n")

def main():
    args = parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Cập nhật và lưu lại file config tạm thời
    cfg = auto_detect_kaggle_paths(cfg, custom_weights=args.weights, custom_raw_data=args.raw_data)
    
    temp_cfg_path = "/kaggle/working/active_config_kaggle.yaml" if os.path.exists("/kaggle") else "./active_config_kaggle.yaml"
    with open(temp_cfg_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    mode = args.mode
    print(f"\n--- Bắt đầu thực thi kịch bản (Mode: {mode}) ---")

    if mode in ["all", "pipeline"]:
        step_data_pipeline(cfg, temp_cfg_path)
    if mode in ["all", "calib", "calib_eval"]:
        step_calibrate(cfg, temp_cfg_path)
    if mode in ["all", "eval", "calib_eval"]:
        step_evaluate_reid(cfg, temp_cfg_path)
    if mode in ["all", "infer"]:
        step_infer_sequence(cfg, temp_cfg_path, specific_seq=args.seq)
    if mode in ["all", "robustness"]:
        step_robustness(cfg, temp_cfg_path)

    print_final_summary(cfg)

if __name__ == "__main__":
    main()
