import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create figure and axes
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Function to draw a box with text
def draw_box(x, y, width, height, color, label, sublabel=""):
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                                  linewidth=2, edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    plt.text(x + width/2, y + height/2 + 0.2, label, ha='center', va='center', fontsize=11, fontweight='bold')
    if sublabel:
        plt.text(x + width/2, y + height/2 - 0.2, sublabel, ha='center', va='center', fontsize=9, style='italic')

# Function to draw arrow
def draw_arrow(x_start, y_start, x_end, y_end):
    ax.annotate("", xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", lw=2, color='black'))

# --- SECTION 1: USER TIER (LEFT) ---
draw_box(0.5, 5, 2, 1.5, '#FFE0B2', "End User", "(Analyst/Laptop)")

# Arrow: User -> Frontend
draw_arrow(2.6, 5.75, 4, 5.75)
plt.text(3.3, 5.9, "HTTPS\nInput", ha='center', fontsize=9)

# --- SECTION 2: DOCKER CONTAINER (CENTER) ---
# Outer Container Box
container = patches.Rectangle((4, 1), 5, 6.5, linewidth=2, edgecolor='#0277BD', facecolor='#E1F5FE', linestyle='--')
ax.add_patch(container)
plt.text(6.5, 7.2, "Docker Container (Render)", ha='center', fontsize=12, fontweight='bold', color='#01579B')

# Frontend
draw_box(4.5, 5, 4, 1, '#B3E5FC', "Streamlit Frontend", "Port: 8501")

# Arrow: Frontend -> Backend
draw_arrow(6.5, 5, 6.5, 4)
plt.text(6.8, 4.5, "JSON\nPayload", ha='left', fontsize=8)

# Backend
draw_box(4.5, 2.5, 4, 1.5, '#81D4FA', "FastAPI Backend", "Port: 8000")

# Model Registry (Inside Backend logic effectively)
draw_box(4.8, 1.2, 3.4, 0.8, '#FFF9C4', "Model Registry", "(.pkl files)")

# Arrow: Backend -> Model
draw_arrow(6.5, 2.5, 6.5, 2.1)

# --- SECTION 3: MLOps TIER (RIGHT) ---
# GitHub Actions
draw_box(9.5, 5, 2, 1.5, '#C8E6C9', "GitHub Actions", "(CI/CD Pipeline)")

# Arrow: GitHub -> Docker
draw_arrow(9.5, 5.75, 9, 5.75)
plt.text(9.25, 6, "Deploy", ha='center', fontsize=9)

# Prefect
draw_box(9.5, 2.5, 2, 1.5, '#E1BEE7', "Prefect", "(Orchestration)")

# Arrow: Prefect -> Model Registry
draw_arrow(9.5, 3.25, 8.3, 1.6)
plt.text(9.3, 2.2, "Retrain &\nUpdate", ha='center', fontsize=9)

# Title
plt.text(6, 0.2, "Fig 1. System Architecture Diagram", ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('system_architecture.png', dpi=300)
print("System Architecture diagram saved as 'system_architecture.png'")
plt.show()