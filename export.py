"""
Export trained model to ONNX format for deployment on Jetson.
"""

import argparse
import os
import torch
import torch.onnx

from model import QualityScorer


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    backbone_name: str = "efficientnet_b4",
    input_size: int = 448,
    opset_version: int = 11,
    dynamic_batch: bool = False,
    simplify: bool = True
):
    """
    Export a trained model to ONNX format.

    Args:
        checkpoint_path: Path to trained model checkpoint
        output_path: Output ONNX file path
        backbone_name: Backbone architecture name
        input_size: Input image size
        opset_version: ONNX opset version
        dynamic_batch: Whether to allow dynamic batch size
        simplify: Whether to simplify the ONNX model
    """
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Try to read config from checkpoint
    if 'config' in checkpoint:
        backbone_name = checkpoint['config'].get('backbone_name', backbone_name)
        input_size = checkpoint['config'].get('input_size', input_size)
        print(f"Using config from checkpoint: backbone={backbone_name}, input_size={input_size}")

    # Create model
    model = QualityScorer(backbone_name=backbone_name, pretrained=False)

    # Load weights
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # Dynamic axes for batch size
    if dynamic_batch:
        dynamic_axes = {
            'image': {0: 'batch_size'},
            'score': {0: 'batch_size'}
        }
    else:
        dynamic_axes = None

    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['image'],
        output_names=['score'],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True
    )

    print(f"ONNX model saved to {output_path}")

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed!")

        # Print model info
        print(f"\nModel inputs:")
        for inp in onnx_model.graph.input:
            print(f"  {inp.name}: {[dim.dim_value for dim in inp.type.tensor_type.shape.dim]}")

        print(f"\nModel outputs:")
        for out in onnx_model.graph.output:
            print(f"  {out.name}: {[dim.dim_value for dim in out.type.tensor_type.shape.dim]}")

    except ImportError:
        print("onnx package not installed, skipping validation")

    # Optionally simplify the model
    if simplify:
        try:
            import onnxsim
            print("\nSimplifying ONNX model...")
            onnx_model = onnx.load(output_path)
            model_simp, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(model_simp, output_path)
                print("Model simplified successfully!")
            else:
                print("Simplification check failed, keeping original model")
        except ImportError:
            print("onnxsim package not installed, skipping simplification")

    # Print TensorRT conversion command
    print(f"\n{'='*60}")
    print("To convert to TensorRT on Jetson, run:")
    print(f"  trtexec --onnx={output_path} --saveEngine=model.trt --fp16")
    print(f"\nFor best performance on Jetson:")
    print(f"  trtexec --onnx={output_path} --saveEngine=model.trt --fp16 --workspace=1024")
    print(f"{'='*60}")


def verify_onnx_inference(
    onnx_path: str,
    pytorch_checkpoint: str,
    backbone_name: str = "efficientnet_b4",
    input_size: int = 448
):
    """
    Verify that ONNX model produces same outputs as PyTorch model.
    """
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("onnxruntime not installed, skipping verification")
        return

    # Load PyTorch model
    checkpoint = torch.load(pytorch_checkpoint, map_location='cpu')
    model = QualityScorer(backbone_name=backbone_name, pretrained=False)

    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Create ONNX session
    ort_session = ort.InferenceSession(onnx_path)

    # Test with random input
    test_input = torch.randn(1, 3, input_size, input_size)

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = model(test_input).numpy()

    # ONNX inference
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
    onnx_output = ort_session.run(None, ort_inputs)[0]

    # Compare outputs
    diff = np.abs(pytorch_output - onnx_output).max()
    print(f"\nVerification:")
    print(f"  PyTorch output: {pytorch_output.flatten()}")
    print(f"  ONNX output: {onnx_output.flatten()}")
    print(f"  Max difference: {diff:.6f}")

    if diff < 1e-4:
        print("  Status: PASSED")
    else:
        print("  Status: WARNING - outputs differ")


def main():
    parser = argparse.ArgumentParser(description='Export model to ONNX')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='model.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--backbone', type=str, default='efficientnet_b4',
                        help='Backbone architecture')
    parser.add_argument('--input-size', type=int, default=448,
                        help='Input image size')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX opset version')
    parser.add_argument('--dynamic-batch', action='store_true',
                        help='Allow dynamic batch size')
    parser.add_argument('--no-simplify', action='store_true',
                        help='Skip ONNX simplification')
    parser.add_argument('--verify', action='store_true',
                        help='Verify ONNX output matches PyTorch')

    args = parser.parse_args()

    # Export
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        backbone_name=args.backbone,
        input_size=args.input_size,
        opset_version=args.opset,
        dynamic_batch=args.dynamic_batch,
        simplify=not args.no_simplify
    )

    # Optionally verify
    if args.verify:
        verify_onnx_inference(
            onnx_path=args.output,
            pytorch_checkpoint=args.checkpoint,
            backbone_name=args.backbone,
            input_size=args.input_size
        )


if __name__ == '__main__':
    main()
