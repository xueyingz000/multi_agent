import ifcopenshell
import sys
import os
import logging
from agents.area_calculation_agent import AreaCalculationAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_debug():
    file_path = "/Users/zhuxueying/ifc/ifc_files/academic b.ifc"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading {file_path}...")
    ifc_file = ifcopenshell.open(file_path)
    
    agent = AreaCalculationAgent()
    
    # Mock data for rules and alignment since we just want to trigger the geometry calc
    rules_data = {"region": "Shanghai"}
    alignment_data = []
    
    print("Running Area Calculation Agent...")
    report = agent.calculate(ifc_file, rules_data, alignment_data)
    
    print("\nCheck the 'debug_viz' folder for images.")

if __name__ == "__main__":
    run_debug()
