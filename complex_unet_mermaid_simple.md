# Complex U-Net Architecture - Simplified View

```mermaid
%%{init: {'theme':'dark', 'themeVariables': { 'fontSize':'18px', 'fontFamily':'monospace'}}}%%
graph TB
    subgraph Input["<b>Input Layer</b>"]
        I1["<b>(8,4,2,12)</b>"]
        I2["<b>Reshape → (32,2,12)</b>"]
    end
    
    subgraph Encoder["<b>Encoder (Downsampling)</b>"]
        E1["<b>Enc-1: (32,32,12)</b>"]
        E2["<b>Enc-2: (32,64,6)</b>"]
        E3["<b>Enc-3: (32,128,3)</b>"]
    end
    
    subgraph Bottleneck["<b>Bottleneck</b>"]
        B1["<b>(32,256,1)</b>"]
    end
    
    subgraph Decoder["<b>Decoder (Upsampling)</b>"]
        D1["<b>Dec-1: (32,128,3)</b>"]
        D2["<b>Dec-2: (32,64,6)</b>"]
        D3["<b>Dec-3: (32,32,12)</b>"]
    end
    
    subgraph Output["<b>Output Layer</b>"]
        O1["<b>Conv: (32,1,12)</b>"]
        O2["<b>Reshape → (8,4,1,12)</b>"]
    end
    
    I1 ==> I2
    I2 ==> E1
    E1 ==> E2
    E2 ==> E3
    E3 ==> B1
    B1 ==> D1
    D1 ==> D2
    D2 ==> D3
    D3 ==> O1
    O1 ==> O2
    
    E1 -.->|<b>Skip</b>| D3
    E2 -.->|<b>Skip</b>| D2
    E3 -.->|<b>Skip</b>| D1
    
    %% High contrast colors for dark theme
    style I1 fill:#2E7D32,stroke:#4CAF50,stroke-width:3px,color:#E8F5E9
    style I2 fill:#2E7D32,stroke:#4CAF50,stroke-width:3px,color:#E8F5E9
    style E1 fill:#1565C0,stroke:#42A5F5,stroke-width:3px,color:#E3F2FD
    style E2 fill:#1565C0,stroke:#42A5F5,stroke-width:3px,color:#E3F2FD
    style E3 fill:#1565C0,stroke:#42A5F5,stroke-width:3px,color:#E3F2FD
    style B1 fill:#F57C00,stroke:#FFB74D,stroke-width:3px,color:#FFF3E0
    style D1 fill:#C2185B,stroke:#F06292,stroke-width:3px,color:#FCE4EC
    style D2 fill:#C2185B,stroke:#F06292,stroke-width:3px,color:#FCE4EC
    style D3 fill:#C2185B,stroke:#F06292,stroke-width:3px,color:#FCE4EC
    style O1 fill:#7B1FA2,stroke:#BA68C8,stroke-width:3px,color:#F3E5F5
    style O2 fill:#7B1FA2,stroke:#BA68C8,stroke-width:3px,color:#F3E5F5
```

**Simplified Architecture Overview**

This diagram shows the high-level structure:
- **Input**: (8,4,2,12) → Reshape → (32,2,12)
- **Encoder**: 3 levels of downsampling (12→6→3→1)
- **Bottleneck**: Deepest representation at (32,256,1)
- **Decoder**: 3 levels of upsampling (1→3→6→12) with skip connections
- **Output**: (32,1,12) → Reshape → (8,4,1,12)

Each port independently processed throughout the network.
