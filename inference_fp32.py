import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite
from gfpgan import GFPGANer
import gc

# ------------------------ FP32-only global settings ------------------------
# Disable half/amp and TF32 to ensure true FP32 math on supported GPUs.
# prefer highest-precision FP32 kernels
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False  # FP32 only
    torch.backends.cudnn.allow_tf32 = False        # FP32 only

# python inference_fp32.py -i inputs/whole_imgs -o results -v 1.4 -s 2 --bg_upsampler realesrgan --bg_tile 128


def ensure_dir(path: str):
    """Create parent folder for a file path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def main():
    """Inference demo for GFPGAN (GPU + alpha-safe, forced FP32)."""

    # ------------------------ device ------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_half = False  # FP32: force no half on both CUDA and CPU
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True  # speed autotuner for conv shapes

    print("=" * 40)
    print("GFPGAN Inference")
    print(
        f"Device: {device} | half: {use_half} | TF32: {False if device=='cuda' else 'N/A'}")
    print("=" * 40)

    # ------------------------ args ------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs/whole_imgs',
                        help='Input image or folder. Default: inputs/whole_imgs')
    parser.add_argument('-o', '--output', type=str, default='results',
                        help='Output folder. Default: results')
    parser.add_argument('-v', '--version', type=str, default='1.3',
                        help='GFPGAN model version. Options: 1 | 1.2 | 1.3 | 1.4 | RestoreFormer. Default: 1.3')
    parser.add_argument('-s', '--upscale', type=int, default=2,
                        help='Final upsampling scale. Default: 2')
    parser.add_argument('--bg_upsampler', type=str, default='realesrgan',
                        help='Background upsampler. Default: realesrgan')
    parser.add_argument('--bg_tile', type=int, default=200,
                        help='Tile size for background upsampler, 0 = no tile. Default: 200 (use smaller on low VRAM)')
    parser.add_argument('--suffix', type=str, default=None,
                        help='Suffix of the restored faces')
    parser.add_argument('--only_center_face', action='store_true',
                        help='Only restore the center face')
    parser.add_argument('--aligned', action='store_true',
                        help='Input are aligned faces (do NOT use for whole photos).')
    parser.add_argument('--ext', type=str, default='auto',
                        help='Image extension: auto | jpg | png. Default: auto')
    parser.add_argument('-w', '--weight', type=float, default=0.5,
                        help='Adjustable blending weight. Default: 0.5')
    args = parser.parse_args()

    # ------------------------ input & output ------------------------
    if args.input.endswith('/'):
        args.input = args.input[:-1]
    if os.path.isfile(args.input):
        img_list = [args.input]
    else:
        img_list = sorted(glob.glob(os.path.join(args.input, '*')))
    os.makedirs(args.output, exist_ok=True)

    # ------------------------ background upsampler ------------------------
    if args.bg_upsampler == 'realesrgan':
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        # Low-VRAM background model (x2); change to scale=4/x4plus if you prefer.
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=model,
            tile=args.bg_tile,
            tile_pad=10,
            pre_pad=0,
            half=False,        # FP32: absolutely disable FP16
            device=device      # ensure it uses CUDA if available
        )
        print(
            f"[GFPGAN] BG upsampler: RealESRGANer on {device} | tile={args.bg_tile} | half=False (FP32)")
    else:
        bg_upsampler = None
        print("[GFPGAN] BG upsampler: None (faces only)")

    # ------------------------ choose GFPGAN model ------------------------
    if args.version == '1':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'GFPGANv1'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
    elif args.version == '1.2':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANCleanv1-NoCE-C2'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
    elif args.version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    elif args.version == '1.4':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif args.version == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    else:
        raise ValueError(f'Wrong model version {args.version}.')

    # determine model paths
    model_path = os.path.join(
        'experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = url  # download pre-trained from url

    # ------------------------ GFPGAN restorer ------------------------
    restorer = GFPGANer(
        model_path=model_path,
        upscale=args.upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler,
        device=device
    )

    # ------------------------ restore ------------------------
    with torch.inference_mode():  # FP32 safe; no autocast anywhere
        for img_path in img_list:
            img_name = os.path.basename(img_path)
            print(f'Processing {img_name} ...')
            basename, ext = os.path.splitext(img_name)

            # Read image (alpha-safe)
            input_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if input_img is None:
                print(f'[WARN] Failed to read {img_path}')
                continue

            alpha = None
            if input_img.ndim == 3 and input_img.shape[2] == 4:
                alpha = input_img[:, :, 3]
                bgr_input = input_img[:, :, :3]
            else:
                bgr_input = input_img

            # restore faces and background
            cropped_faces, restored_faces, restored_img = restorer.enhance(
                bgr_input,
                has_aligned=args.aligned,
                only_center_face=args.only_center_face,
                paste_back=True,
                weight=args.weight
            )

            if restored_img is None and not restored_faces:
                print(f'[INFO] No faces restored in: {img_name}')
                # ---- cleanup for this image ----
                del input_img
                if 'bgr_input' in locals():
                    del bgr_input
                if 'alpha' in locals():
                    del alpha
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue

            # save faces
            for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
                save_crop_path = os.path.join(
                    args.output, 'cropped_faces', f'{basename}_{idx:02d}.png')
                ensure_dir(save_crop_path)
                imwrite(cropped_face, save_crop_path)

                if args.suffix is not None:
                    save_face_name = f'{basename}_{idx:02d}_{args.suffix}.png'
                else:
                    save_face_name = f'{basename}_{idx:02d}.png'
                save_restore_path = os.path.join(
                    args.output, 'restored_faces', save_face_name)
                ensure_dir(save_restore_path)
                imwrite(restored_face, save_restore_path)

            # save restored full image
            if restored_img is not None:
                # re-attach alpha if possible
                force_png = False
                if alpha is not None and restored_img.shape[:2] == alpha.shape[:2]:
                    restored_img = np.dstack((restored_img, alpha))
                    force_png = True

                extension = ext[1:] if args.ext == 'auto' else args.ext
                if not extension:
                    extension = 'png'
                if force_png and extension.lower() != 'png':
                    extension = 'png'

                if args.suffix is not None:
                    save_restore_path = os.path.join(
                        args.output, 'restored_imgs', f'{basename}_{args.suffix}.{extension}')
                else:
                    save_restore_path = os.path.join(
                        args.output, 'restored_imgs', f'{basename}.{extension}')
                ensure_dir(save_restore_path)
                imwrite(restored_img, save_restore_path)

            # ---- cleanup for this image (frees VRAM + RAM) ----
            del cropped_faces, restored_faces
            if restored_img is not None:
                del restored_img
            del input_img
            if 'bgr_input' in locals():
                del bgr_input
            if 'cmp_img' in locals():
                del cmp_img
            if 'alpha' in locals():
                del alpha
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print(f'Results are in the [{args.output}] folder.')


if __name__ == '__main__':
    main()
