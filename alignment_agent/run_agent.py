#!/usr/bin/env python3
"""Main entry point for IFC Semantic Agent."""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from core import IFCSemanticAgent
from utils import get_logger, setup_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="IFC Semantic Agent - Intelligent IFC-Regulatory Alignment System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python run_agent.py --interactive
  
  # Process single query
  python run_agent.py --query "如何将IfcWall映射到监管要求？"
  
  # Process with IFC file
  python run_agent.py --query "分析IFC模型合规性" --ifc-file model.ifc
  
  # Process with regulatory text
  python run_agent.py --query "检查建筑规范" --text-file regulations.txt
  
  # Run examples
  python run_agent.py --example basic
  python run_agent.py --example advanced
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Configuration file path (default: config.yaml)"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Query to process"
    )
    
    parser.add_argument(
        "--ifc-file",
        type=str,
        help="IFC file path to analyze"
    )
    
    parser.add_argument(
        "--text-file",
        type=str,
        help="Regulatory text file path"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path for results"
    )
    
    parser.add_argument(
        "--format",
        choices=["json", "yaml", "txt"],
        default="txt",
        help="Output format (default: txt)"
    )
    
    parser.add_argument(
        "--example",
        choices=["basic", "advanced"],
        help="Run predefined examples"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser.parse_args()


def load_file_content(file_path: str) -> Optional[str]:
    """Load content from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


def save_results(results: Dict[str, Any], output_path: str, format_type: str):
    """Save results to file."""
    try:
        if format_type == "json":
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        elif format_type == "yaml":
            import yaml
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(results, f, allow_unicode=True, default_flow_style=False)
        else:  # txt format
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"IFC Semantic Agent Results\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Query: {results.get('query', 'N/A')}\n")
                f.write(f"Final Answer: {results.get('final_answer', 'N/A')}\n")
                f.write(f"Confidence Score: {results.get('confidence_score', 0):.2f}\n")
                f.write(f"Total Steps: {results.get('total_steps', 0)}\n")
                f.write(f"Execution Time: {results.get('execution_time', 0):.2f}s\n\n")
                
                if 'react_steps' in results:
                    f.write(f"Reasoning Steps:\n")
                    f.write(f"{'-'*30}\n")
                    for i, step in enumerate(results['react_steps'], 1):
                        f.write(f"Step {i}: {step.get('thought', {}).get('reasoning_type', 'Unknown')}\n")
                        f.write(f"Thought: {step.get('thought', {}).get('content', 'N/A')}\n")
                        f.write(f"Action: {step.get('action', {}).get('action_type', 'N/A')}\n")
                        f.write(f"Success: {'Yes' if step.get('success', False) else 'No'}\n\n")
        
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")


def run_interactive_mode(agent: IFCSemanticAgent):
    """Run agent in interactive mode."""
    print("\n" + "="*60)
    print("IFC Semantic Agent - Interactive Mode")
    print("="*60)
    print("Enter your queries about IFC-regulatory alignment.")
    print("Commands:")
    print("  /help    - Show help")
    print("  /status  - Show agent status")
    print("  /reset   - Reset agent")
    print("  /quit    - Exit")
    print("-"*60)
    
    while True:
        try:
            query = input("\n> ").strip()
            
            if not query:
                continue
            
            if query == "/quit":
                print("Goodbye!")
                break
            elif query == "/help":
                print("\nAvailable commands:")
                print("  /help    - Show this help message")
                print("  /status  - Show current agent status")
                print("  /reset   - Reset agent state")
                print("  /quit    - Exit interactive mode")
                print("\nOr enter any query about IFC-regulatory alignment.")
                continue
            elif query == "/status":
                state = agent.get_agent_state()
                print(f"\nAgent Status:")
                print(f"  Current Step: {state['current_step']}")
                print(f"  History Length: {state['react_history_length']}")
                print(f"  Overall Confidence: {state['overall_confidence']:.2f}")
                print(f"  Available Actions: {len(state['available_actions'])}")
                continue
            elif query == "/reset":
                agent.reset_agent()
                print("Agent reset successfully.")
                continue
            
            print(f"\nProcessing query: {query}")
            print("-"*40)
            
            response = agent.process_query(
                query=query,
                ifc_data=None,
                regulatory_text=None
            )
            
            print(f"\nAnswer: {response.final_answer}")
            print(f"Confidence: {response.confidence_score:.2f}")
            print(f"Steps: {response.total_steps}")
            print(f"Time: {response.execution_time:.2f}s")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_single_query(agent: IFCSemanticAgent, query: str, ifc_file: Optional[str] = None, text_file: Optional[str] = None):
    """Run a single query."""
    print(f"\nProcessing query: {query}")
    print("-"*50)
    
    # Load IFC data if provided
    ifc_data = None
    if ifc_file:
        print(f"Loading IFC file: {ifc_file}")
        file_ext = Path(ifc_file).suffix.lower()
        if file_ext == '.ifc':
            # Process IFC file using IFCProcessor
            from data_processing import IFCProcessor
            ifc_processor = IFCProcessor()
            try:
                ifc_data = ifc_processor.process_ifc_file(ifc_file)
                print(f"IFC file processed: {ifc_data.get('entity_count', 0)} entities extracted")
            except Exception as e:
                print(f"Error processing IFC file: {e}")
                # Fallback to raw content
                ifc_content = load_file_content(ifc_file)
                if ifc_content:
                    ifc_data = {"raw_content": ifc_content}
        else:
            # For other formats, treat as text content
            ifc_content = load_file_content(ifc_file)
            if ifc_content:
                ifc_data = {"raw_content": ifc_content}
    
    # Load regulatory text if provided
    regulatory_text = None
    if text_file:
        print(f"Loading text file: {text_file}")
        file_ext = Path(text_file).suffix.lower()
        if file_ext == '.json':
            # Process JSON regulatory file using TextProcessor
            from data_processing import TextProcessor
            text_processor = TextProcessor()
            try:
                text_data = text_processor.process_text_file(text_file)
                regulatory_text = text_data.get('cleaned_text', '')
                print(f"JSON regulatory file processed: {text_data.get('regulation_count', 0)} regulations, {text_data.get('chunk_count', 0)} chunks")
            except Exception as e:
                print(f"Error processing JSON regulatory file: {e}")
                # Fallback to raw content
                regulatory_text = load_file_content(text_file)
        else:
            # For other formats, load as plain text
            regulatory_text = load_file_content(text_file)
    
    # Process query
    response = agent.process_query(
        query=query,
        ifc_data=ifc_data,
        regulatory_text=regulatory_text
    )
    
    # Display results
    print(f"\nResults:")
    print(f"{'='*50}")
    print(f"Final Answer: {response.final_answer}")
    print(f"Confidence Score: {response.confidence_score:.2f}")
    print(f"Total Steps: {response.total_steps}")
    print(f"Execution Time: {response.execution_time:.2f}s")
    
    if response.react_steps:
        print(f"\nReasoning Steps:")
        print(f"{'-'*30}")
        for i, step in enumerate(response.react_steps, 1):
            print(f"Step {i}: {step.thought.reasoning_type}")
            print(f"  Thought: {step.thought.content[:100]}...")
            print(f"  Action: {step.action.action_type.value}")
            print(f"  Success: {'Yes' if step.success else 'No'}")
    
    return {
        "query": query,
        "final_answer": response.final_answer,
        "confidence_score": response.confidence_score,
        "total_steps": response.total_steps,
        "execution_time": response.execution_time,
        "react_steps": [{
            "thought": {
                "reasoning_type": step.thought.reasoning_type,
                "content": step.thought.content
            },
            "action": {
                "action_type": step.action.action_type.value,
                "reasoning": step.action.reasoning
            },
            "success": step.success
        } for step in response.react_steps]
    }


def run_example(example_type: str):
    """Run predefined examples."""
    if example_type == "basic":
        print("Running basic usage example...")
        try:
            from examples.basic_usage import main as run_basic_example
            run_basic_example()
        except ImportError:
            print("Basic example not found. Please ensure examples/basic_usage.py exists.")
    elif example_type == "advanced":
        print("Running advanced alignment example...")
        try:
            from examples.advanced_alignment import main as run_advanced_example
            run_advanced_example()
        except ImportError:
            print("Advanced example not found. Please ensure examples/advanced_alignment.py exists.")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else ("INFO" if args.verbose else "WARNING")
    setup_logger({'level': log_level})
    
    logger = get_logger(__name__)
    logger.info("Starting IFC Semantic Agent")
    
    # Check if running examples
    if args.example:
        run_example(args.example)
        return
    
    # Initialize agent
    try:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Configuration file not found: {config_path}")
            print("Please ensure config.yaml exists or specify a valid config file with --config")
            return
        
        print(f"Initializing IFC Semantic Agent with config: {config_path}")
        agent = IFCSemanticAgent(str(config_path))
        print("Agent initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing agent: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return
    
    # Run based on mode
    try:
        if args.interactive:
            run_interactive_mode(agent)
        elif args.query:
            results = run_single_query(
                agent=agent,
                query=args.query,
                ifc_file=args.ifc_file,
                text_file=args.text_file
            )
            
            # Save results if output path specified
            if args.output:
                save_results(results, args.output, args.format)
        else:
            print("No action specified. Use --interactive, --query, or --example.")
            print("Run 'python run_agent.py --help' for usage information.")
    
    except Exception as e:
        print(f"Error during execution: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
    
    finally:
        logger.info("IFC Semantic Agent session ended")


if __name__ == "__main__":
    main()