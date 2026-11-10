# ReLU Activation Function Visualiser (Streamlit app)
# Run with:  streamlit run relu_app.py
#
# What this app does:
#   1. Shows the formula of the ReLU activation function
#   2. Plots the function and its derivative (gradient)
#   3. Lets you test one neuron:  z = w.x + b  ->  output = f(z)

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# ---------------------------------------------------------------
# 1. MATHS: the activation function and its derivative
# ---------------------------------------------------------------
def activation(z):
    """ReLU: returns z if z > 0, otherwise 0."""
    return np.maximum(0, z)


def derivative(z):
    """Slope of ReLU: 1 for positive z, 0 otherwise."""
    return np.where(z > 0, 1.0, 0.0)


# ---------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------------
def neuron_output(inputs, weights, bias):
    """One neuron: weighted sum plus bias (z), then the activation function."""
    z = float(np.dot(inputs, weights) + bias)
    return z, float(activation(z))


def plot_function(z_min, z_max, z_value, show_derivative=True):
    """Draw the activation function (and optionally its derivative)."""
    z = np.linspace(z_min, z_max, 400)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z, activation(z), linewidth=2, label="ReLU  f(z)")
    if show_derivative:
        ax.plot(z, derivative(z), "--", label="Derivative  f'(z)")
    ax.scatter([z_value], [activation(z_value)], color="red", zorder=5,
               label="Selected input")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("z  (weighted sum + bias)")
    ax.set_ylabel("Output")
    ax.set_title("ReLU Activation Function")
    ax.grid(True)
    ax.legend()
    return fig


# ---------------------------------------------------------------
# 3. PAGE LAYOUT
# ---------------------------------------------------------------
st.set_page_config(page_title="ReLU Activation Function", layout="centered")
st.title("ReLU (Rectified Linear Unit) Activation Function")
st.write("ReLU allows only positive values to pass through it. Negative values are mapped to zero.")
st.latex(r"f(z) = \max(0, z)")

# Sidebar controls
st.sidebar.header("Controls")
z_min, z_max = st.sidebar.slider("Range of z (x-axis)", -10.0, 10.0,
                                 (-5.0, 5.0), step=0.5)
z_value = st.sidebar.slider("Pick an input value z", -10.0, 10.0,
                            2.0, step=0.01)
show_derivative = st.sidebar.checkbox("Show derivative (gradient)", value=True)

if z_min >= z_max:
    st.error("The minimum of the range must be smaller than the maximum.")
    st.stop()

tab_graph, tab_neuron, tab_table, tab_notes = st.tabs(
    ["Graph", "Neuron calculator", "Table of values", "Key points"])

# ---- Tab 1: graph ----
with tab_graph:
    fig = plot_function(z_min, z_max, z_value, show_derivative)
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)
    col1.metric("Output f(z)", f"{float(activation(z_value)):.4f}")
    col2.metric("Derivative f'(z)", f"{float(derivative(z_value)):.4f}")

# ---- Tab 2: one neuron, step by step ----
with tab_neuron:
    st.write("Change the inputs (x), weights (w) and bias (b). "
             "The neuron computes z = w1*x1 + w2*x2 + ... + b and then applies "
             "the ReLU function.")
    default_inputs = [1.0, 0.0, 1.0]
    default_weights = [0.5, -0.2, 0.3]
    inputs, weights = [], []
    cols = st.columns(len(default_inputs))
    for i, col in enumerate(cols):
        inputs.append(col.number_input(f"x{i + 1}", value=default_inputs[i],
                                       step=0.1, key=f"x{i}"))
        weights.append(col.number_input(f"w{i + 1}", value=default_weights[i],
                                        step=0.1, key=f"w{i}"))
    bias = st.number_input("Bias b", value=0.0, step=0.1)

    z_neuron, out_neuron = neuron_output(inputs, weights, bias)
    c1, c2 = st.columns(2)
    c1.metric("Weighted sum z", f"{z_neuron:.4f}")
    c2.metric("Neuron output f(z)", f"{out_neuron:.4f}")
    st.caption("Lecture example (hidden neuron H1): inputs [1, 0, 1] and weights [0.5, -0.2, 0.3] give z = 0.8, so ReLU(0.8) = 0.8. For H2 (weights [0.4, 0.1, -0.5]) z = -0.1, so ReLU(-0.1) = 0.")

# ---- Tab 3: table of values ----
with tab_table:
    samples = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]
    st.table({
        "z": samples,
        "f(z)": [round(float(activation(s)), 4) for s in samples],
        "f'(z)": [round(float(derivative(s)), 4) for s in samples],
    })

# ---- Tab 4: key points ----
with tab_notes:
    st.markdown(
        "- **Range:** 0 to infinity (no upper limit).\n"
        "- **Negative inputs** give 0; positive inputs pass through unchanged.\n"
        "- **Non-linear**, so the network can learn complex patterns.\n"
        "- **Very cheap to compute**; widely used in hidden layers.\n"
        "- **Weakness:** a neuron that always receives negative input outputs 0 and stops learning (the dying ReLU problem)."
    )
