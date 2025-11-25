"""
Complex U-Net Architecture - Mermaid Diagram Generator
Generate flowchart in Mermaid format for better rendering
"""

def generate_mermaid_diagram():
    """Generate Mermaid flowchart for Complex U-Net"""
    
    B, P, L = 8, 4, 12
    base_ch = 32
    
    mermaid_code = """```mermaid
%%{init: {'theme':'dark', 'themeVariables': { 'fontSize':'16px', 'fontFamily':'monospace'}}}%%
flowchart TD
    %% Styling for dark theme with high contrast
    classDef inputStyle fill:#2E7D32,stroke:#4CAF50,stroke-width:3px,color:#E8F5E9,font-weight:bold
    classDef encoderStyle fill:#1565C0,stroke:#42A5F5,stroke-width:3px,color:#E3F2FD,font-weight:bold
    classDef bottleneckStyle fill:#F57C00,stroke:#FFB74D,stroke-width:3px,color:#FFF3E0,font-weight:bold
    classDef decoderStyle fill:#C2185B,stroke:#F06292,stroke-width:3px,color:#FCE4EC,font-weight:bold
    classDef outputStyle fill:#7B1FA2,stroke:#BA68C8,stroke-width:3px,color:#F3E5F5,font-weight:bold
    
    %% Input Layer
    Input["<b>Input</b><br/>(8,4,2,12)<br/>Channel + Position Encoding"]:::inputStyle
    Reshape1["<b>Reshape</b><br/>(32,2,12)<br/>Merge ports to batch"]:::inputStyle
    
    %% Encoder Path
    Enc1["<b>Encoder-1</b><br/>(32,32,12)<br/>ResBlock + Attention"]:::encoderStyle
    Down1["<b>DownSample-1</b><br/>(32,32,6)<br/>Conv1d k=2, s=2"]:::encoderStyle
    
    Enc2["<b>Encoder-2</b><br/>(32,64,6)<br/>ResBlock + Attention"]:::encoderStyle
    Down2["<b>DownSample-2</b><br/>(32,64,3)<br/>Conv1d k=2, s=2"]:::encoderStyle
    
    Enc3["<b>Encoder-3</b><br/>(32,128,3)<br/>ResBlock + Attention"]:::encoderStyle
    Down3["<b>DownSample-3</b><br/>(32,128,1)<br/>Conv1d k=2, s=2"]:::encoderStyle
    
    %% Bottleneck
    Bottleneck["<b>Bottleneck</b><br/>(32,256,1)<br/>ResBlock + Attention"]:::bottleneckStyle
    
    %% Decoder Path
    Up1["<b>UpSample-1</b><br/>(32,128,3)<br/>ConvTranspose1d k=2, s=2"]:::decoderStyle
    Cat1["<b>Concat-1</b><br/>(32,256,3)<br/>Cat with Enc3"]:::decoderStyle
    Dec1["<b>Decoder-1</b><br/>(32,128,3)<br/>ResBlock + Attention"]:::decoderStyle
    
    Up2["<b>UpSample-2</b><br/>(32,64,6)<br/>ConvTranspose1d k=2, s=2"]:::decoderStyle
    Cat2["<b>Concat-2</b><br/>(32,128,6)<br/>Cat with Enc2"]:::decoderStyle
    Dec2["<b>Decoder-2</b><br/>(32,64,6)<br/>ResBlock + Attention"]:::decoderStyle
    
    Up3["<b>UpSample-3</b><br/>(32,32,12)<br/>ConvTranspose1d k=2, s=2"]:::decoderStyle
    Cat3["<b>Concat-3</b><br/>(32,64,12)<br/>Cat with Enc1"]:::decoderStyle
    Dec3["<b>Decoder-3</b><br/>(32,32,12)<br/>ResBlock + Attention"]:::decoderStyle
    
    %% Output Layer
    OutConv["<b>Output Conv</b><br/>(32,1,12)<br/>Conv1d k=1"]:::outputStyle
    Reshape2["<b>Reshape</b><br/>(8,4,1,12)<br/>Restore ports dimension"]:::outputStyle
    Output["<b>Final Output</b><br/>(8,4,1,12)<br/>Denoised Channel Estimate"]:::outputStyle
    
    %% Flow - Main path with thick arrows
    Input ==> Reshape1
    Reshape1 ==> Enc1
    Enc1 ==> Down1
    Down1 ==> Enc2
    Enc2 ==> Down2
    Down2 ==> Enc3
    Enc3 ==> Down3
    Down3 ==> Bottleneck
    
    Bottleneck ==> Up1
    Up1 ==> Cat1
    Enc3 -.->|<b>Skip</b>| Cat1
    Cat1 ==> Dec1
    
    Dec1 ==> Up2
    Up2 ==> Cat2
    Enc2 -.->|<b>Skip</b>| Cat2
    Cat2 ==> Dec2
    
    Dec2 ==> Up3
    Up3 ==> Cat3
    Enc1 -.->|<b>Skip</b>| Cat3
    Cat3 ==> Dec3
    
    Dec3 ==> OutConv
    OutConv ==> Reshape2
    Reshape2 ==> Output
```

**Network Parameters:**
- Batch Size: 8
- Num Ports: 4
- Sequence Length: 12
- Input Channels: 2 (Channel estimate + Position encoding)
- Output Channels: 1 (Residual)
- Base Channels: 32
- Network Depth: 3
- Attention: Enabled (ComplexAttention)
- Activation: ComplexModReLU
- Normalization: ComplexBatchNorm1d

**Key Features:**
- Full complex number support
- Encoder-decoder structure with skip connections
- Residual blocks + Attention at each layer
- Adaptive to different sequence lengths (12-816)
- Independent position encoding for each port

**Data Flow:**
1. Input: (B, P, C, L) = (8, 4, 2, 12)
2. Reshape: Merge ports to batch -> (32, 2, 12)
3. Encoder: Progressive downsampling and feature extraction
4. Bottleneck: Deepest feature representation
5. Decoder: Progressive upsampling with skip connections
6. Output: (32, 1, 12) -> Reshape -> (8, 4, 1, 12)

**Each Port Gets Independent Output:**
- Port 0, 1, 2, 3 each produces a 12-length complex sequence
- Total output shape: (8 batches, 4 ports, 1 channel, 12 length)
"""
    
    return mermaid_code

def generate_simplified_mermaid():
    """Generate a simplified version for better readability"""
    
    simplified = """```mermaid
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
"""
    
    return simplified

if __name__ == "__main__":
    # Generate detailed Mermaid diagram
    detailed_diagram = generate_mermaid_diagram()
    
    # Generate simplified version
    simplified_diagram = generate_simplified_mermaid()
    
    # Save to markdown files
    with open('complex_unet_mermaid_detailed.md', 'w', encoding='utf-8') as f:
        f.write("# Complex U-Net Architecture - Detailed View\n\n")
        f.write(detailed_diagram)
    
    with open('complex_unet_mermaid_simple.md', 'w', encoding='utf-8') as f:
        f.write("# Complex U-Net Architecture - Simplified View\n\n")
        f.write(simplified_diagram)
    
    # Also save combined version
    with open('complex_unet_architecture.md', 'w', encoding='utf-8') as f:
        f.write("# Complex U-Net Architecture Documentation\n\n")
        f.write("## Table of Contents\n")
        f.write("- [Simplified Overview](#simplified-overview)\n")
        f.write("- [Detailed Architecture](#detailed-architecture)\n\n")
        f.write("---\n\n")
        f.write("## Simplified Overview\n\n")
        f.write(simplified_diagram)
        f.write("\n\n---\n\n")
        f.write("## Detailed Architecture\n\n")
        f.write(detailed_diagram)
    
    print("[OK] Mermaid diagrams generated successfully!")
    print("  - complex_unet_mermaid_detailed.md (Detailed view)")
    print("  - complex_unet_mermaid_simple.md (Simplified view)")
    print("  - complex_unet_architecture.md (Combined documentation)")
    print("\nYou can:")
    print("  1. View these files in VS Code with Mermaid extension")
    print("  2. Copy to GitHub/GitLab (native Mermaid support)")
    print("  3. Use online Mermaid Live Editor: https://mermaid.live")
