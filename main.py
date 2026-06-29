import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def start_server():
    """Starts the FastAPI web server."""
    import uvicorn
    print("Starting FUNDR Analytics Server...")
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)

def run_portfolio():
    """Runs the portfolio generation and analysis logic."""
    from src.portfolio.manager import PortfolioManager
    from src.core.ai_client import AIClient
    
    manager = PortfolioManager()
    ai_client = AIClient(env_path=os.path.join(BASE_DIR, ".env"))
    
    print("\n=== Stage 1: Generating Opportunities ===")
    manager.generate_opportunities()
    
    print("\n=== Stage 2: Analyzing Strategy ===")
    manager.analyze_strategy(ai_client=ai_client)
    
    print("\nPortfolio pipeline complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [server|portfolio]")
        sys.exit(1)
        
    command = sys.argv[1].lower()
    if command == "server":
        start_server()
    elif command == "portfolio":
        run_portfolio()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python main.py [server|portfolio]")
        sys.exit(1)
