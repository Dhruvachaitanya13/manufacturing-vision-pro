import argparse
import sys
from loguru import logger
from src.config import settings
from src.data_gen import AdvancedDataFactory
from src.training import TrainingPipeline

def banner():
    print(f"""
    ===========================================
      🏭 MANUFACTURING VISION PRO v{settings.VERSION}
    ===========================================
    """)

def run_gen(args):
    logger.info(f"Initiating Data Factory (Train: {args.train}, Val: {args.val})")
    factory = AdvancedDataFactory(num_train=args.train, num_val=args.val)
    factory.generate()

def run_train(args):
    logger.info("Initiating Training Pipeline")
    trainer = TrainingPipeline()
    trainer.run()

def run_server(args):
    import uvicorn
    logger.info(f"Starting API Server on Port {args.port}")
    uvicorn.run("src.api:app", host="0.0.0.0", port=args.port, reload=True)

def main():
    banner()
    parser = argparse.ArgumentParser(description="Manufacturing Vision CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Gen Data
    p_gen = subparsers.add_parser("gen", help="Generate Synthetic Dataset")
    p_gen.add_argument("--train", type=int, default=500, help="Training samples")
    p_gen.add_argument("--val", type=int, default=100, help="Validation samples")
    p_gen.set_defaults(func=run_gen)

    # 2. Train
    p_train = subparsers.add_parser("train", help="Train the Model")
    p_train.set_defaults(func=run_train)

    # 3. Serve
    p_serve = subparsers.add_parser("serve", help="Start API Server")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=run_server)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Execution Interrupted by User.")
        sys.exit(0)