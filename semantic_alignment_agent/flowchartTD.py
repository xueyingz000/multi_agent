flowchart TD
    A[Start] --> B{Candidate Vertical Opening<br>Spans ≥ 2 floors?}
    B -- No --> C[Not a Vertical Penetrating Space]
    B -- Yes --> D{Alignment Check<br>Alignment Score ≥ 0.6?}
    
    D -- No, < 0.6 --> E[Classify as: Atrium]
    D -- Yes, ≥ 0.6 --> F{Internal Content Check<br>Stairs/Elevators/MEP Present?}
    
    F -- Yes --> G[Classify as: Functional Shaft]
    F -- No --> H{Edge Enclosure Check<br>What encloses the opening?}
    
    H -- Railing/Handrail --> E
    H -- Wall --> G
    H -- Unknown/None --> I{Size & Shape Check}
    
    I -- Area ≥ 50 AND<br>Aspect Ratio < 2 --> E
    I -- Area ≤ 15 OR<br>Aspect Ratio ≥ 3 --> G
    I -- Otherwise --> J[Default to: Functional Shaft]
    
    E --> K[Output: Atrium<br>Area Counted as 0 per floor]
    C & G & J --> L[Output: Functional Shaft<br>Area Counted per floor]