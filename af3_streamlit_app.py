import os
import re
import json
import numpy as np
import pandas as pd
import zipfile
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import io
from PIL import Image
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from streamlit_molstar import st_molstar

# Function to load JSON data from a file or file object
def extract_data(file_or_path):
    try:
        if isinstance(file_or_path, str):
            with open(file_or_path, 'r') as f:
                data = json.load(f)
        elif hasattr(file_or_path, 'read'):
            # Reset the file pointer to the beginning
            file_or_path.seek(0)
            data = json.load(file_or_path)
        else:
            raise ValueError("Invalid input. Must be a file path or file-like object.")
        return data
    except (ValueError, json.JSONDecodeError, OSError) as e:
        st.error(f"Error reading JSON file: {e}")
        return None

# Helper function to calculate chain lengths from token_res_ids
def calculate_chain_lengths(token_res_ids):
    chain_lengths = []
    current_chain_length = 1
    previous_residue = token_res_ids[0]

    for residue in token_res_ids[1:]:
        if residue == previous_residue + 1:
            current_chain_length += 1
        else:
            chain_lengths.append(current_chain_length)
            current_chain_length = 1
        previous_residue = residue

    chain_lengths.append(current_chain_length)
    return chain_lengths

# Function to plot PAE heatmap and reload it with specific pixel dimensions
# def plot_pae_heatmap(pae_matrix, model_name):
#     fig, ax = plt.subplots(figsize=(6, 6))  # Create the figure with default size

#     cmap = sns.color_palette("Greens", as_cmap=True).reversed()

#     sns.heatmap(pae_matrix, cmap=cmap, square=True, cbar_kws={
#         'label': 'Expected Position Error (Å)',
#         'orientation': 'horizontal',
#         'pad': 0.15,
#         'shrink': 0.65
#     }, ax=ax)

#     ax.set_xlabel("Scored Residue", fontsize=12)
#     ax.set_ylabel("Aligned Residue", fontsize=12)

#     ax.tick_params(axis='both', which='both', length=0)
#     ax.set_xticks(np.arange(0, len(pae_matrix), max(1, len(pae_matrix) // 8)))
#     ax.set_xticklabels(np.arange(1, len(pae_matrix) + 1, max(1, len(pae_matrix) // 8)))
#     ax.set_yticks(np.arange(0, len(pae_matrix), max(1, len(pae_matrix) // 8)))
#     ax.set_yticklabels(np.arange(1, len(pae_matrix) + 1, max(1, len(pae_matrix) // 8)))

#     for _, spine in ax.spines.items():
#         spine.set_visible(True)
#         spine.set_color('black')
#         spine.set_linewidth(1)

#     cbar = ax.collections[0].colorbar
#     cbar.outline.set_edgecolor('black')
#     cbar.outline.set_linewidth(1)

#     cbar.ax.tick_params(length=0)
#     cbar.ax.set_aspect('auto')

#     plt.title(f"{model_name} PAE Heatmap", fontsize=14)

#     # Save the figure in memory as a BytesIO object
#     buf = io.BytesIO()
#     plt.savefig(buf, format="png", bbox_inches="tight")
#     buf.seek(0)  # Move the buffer's position to the beginning

#     # Reload the plot from memory and specify the pixel dimensions
#     image = Image.open(buf)
#     st.image(image, caption=f"{model_name} PAE Heatmap", use_column_width=True)

#     buf.close()  # Close the buffer

def plot_pae_heatmap(pae_matrix, model_name):
    import plotly.graph_objects as go

    num_residues = len(pae_matrix)
    cell_size = 20  # pixels per cell; adjust as needed
    fig_height = num_residues * cell_size

    # Set a maximum figure height to prevent the plot from becoming too tall
    max_height = 800  # Maximum height in pixels; adjust as needed
    if fig_height > max_height:
        fig_height = max_height

    # Create the heatmap trace
    heatmap = go.Heatmap(
        z=pae_matrix,
        colorscale='Greens_r',
        colorbar=dict(
            title='Expected Position Error (Å)',
            orientation='h',
            x=0.5,
            xanchor='center',
            thickness=15,
            len=0.7,
        ),
        zmin=np.min(pae_matrix),
        zmax=np.max(pae_matrix),
        showscale=True,
    )

    # Create the figure
    fig = go.Figure(data=heatmap)

    # Update axes to fix aspect ratio
    fig.update_xaxes(
        tickmode='linear',
        tick0=0,
        dtick=max(1, num_residues // 8),
        showgrid=False,
    )

    fig.update_yaxes(
        tickmode='linear',
        tick0=0,
        dtick=max(1, num_residues // 8),
        showgrid=False,
        scaleanchor="x",   # Tie y-axis to x-axis
        scaleratio=1,
    )

    # Update layout
    fig.update_layout(
        title=f"{model_name} PAE Heatmap",
        xaxis_title="Scored Residue",
        yaxis_title="Aligned Residue",
        autosize=False,
        height=fig_height,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    # Display the figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Function to create a 3D visualization using Mol*
def visualize_structure_with_molstar(cif_file_or_path):
    import tempfile
    import os

    if isinstance(cif_file_or_path, str):
        # If it's a file path, pass it directly to st_molstar
        st.write(f"Loading CIF file from path: {cif_file_or_path}")
        st_molstar(cif_file_or_path, height='600px')
    else:
        # If it's an uploaded file, write it to a temporary file and pass the path
        st.write(f"Loading CIF file from uploaded file: {cif_file_or_path.name}")
        try:
            # Read the content
            content = cif_file_or_path.read()
            # Write to temp file
            with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            # Reset the file pointer
            cif_file_or_path.seek(0)
            st_molstar(tmp_file_path, height='600px')
        finally:
            # Clean up temporary file
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

# Function to display ptm and ipTM averages and ipTM matrix
def display_entity_ptm_iptm_averages_and_matrix(summary_data, model_name):
    chain_ptm = summary_data.get('chain_ptm', [])
    chain_iptm = summary_data.get('chain_iptm', [])

    col1, col2 = st.columns([1, 1], gap="medium")  # Equal column widths for tables

    if chain_ptm and chain_iptm:
        with col1:
            st.write(f"### Entity pTM and ipTM Averages for {model_name}")
            ptm_iptm_df = pd.DataFrame({
                'Chain': [chr(65 + i) for i in range(len(chain_ptm))],
                'chain_ptm': chain_ptm,
                'chain_iptm': chain_iptm
            })
            st.dataframe(ptm_iptm_df)

    chain_pair_iptm = summary_data.get('chain_pair_iptm', [])
    if chain_pair_iptm:
        with col2:
            st.write(f"### ipTM Matrix for {model_name}")
            iptm_matrix_df = pd.DataFrame(chain_pair_iptm, columns=[chr(65 + i) for i in range(len(chain_pair_iptm))])
            st.dataframe(iptm_matrix_df)

# Function to display sequence information and calculate chain lengths
def display_sequence_info(job_request_file, full_data_file):
    job_data = extract_data(job_request_file)
    full_data = extract_data(full_data_file)

    if job_data and 'sequences' in job_data[0]:
        sequence_info = job_data[0]['sequences']
        st.write("### Sequence Information")
        for item in sequence_info:
            if 'proteinChain' in item:
                st.write(f"Type: Protein\nCopies: {item['proteinChain']['count']}\nSequence: {item['proteinChain']['sequence']}")
            if 'dnaSequence' in item:
                st.write(f"Type: DNA\nCopies: {item['dnaSequence']['count']}\nSequence: {item['dnaSequence']['sequence']}")

    if full_data and 'token_res_ids' in full_data:
        token_res_ids = full_data['token_res_ids']
        chain_lengths = calculate_chain_lengths(token_res_ids)
        total_residues = sum(chain_lengths)
        st.write("\n### Chain Lengths")
        st.write(f"Chain lengths: {chain_lengths}")
        st.write(f"Total number of residues: {total_residues}")

# Function to scan for model files in a directory (CIF, JSON)
def scan_model_files(folder_path):
    try:
        models = {}
        base_name = None  # To store the common base name
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                # Skip hidden files
                if file_name.startswith('.'):
                    continue

                # Match job request file to get the base name
                job_request_match = re.match(r'(.*)_job_request\.json', file_name)
                if job_request_match:
                    base_name = job_request_match.group(1)

                model_match = re.match(r'(.*)_model_(\d+)\.cif', file_name)
                summary_match = re.match(r'(.*)_summary_confidences_(\d+)\.json', file_name)
                full_data_match = re.match(r'(.*)_full_data_(\d+)\.json', file_name)

                if model_match:
                    model_base_name = model_match.group(1)
                    index = model_match.group(2)
                    if index not in models:
                        models[index] = {}
                    models[index]['model'] = os.path.join(root, file_name)
                    if not base_name:
                        base_name = model_base_name
                if summary_match:
                    summary_base_name = summary_match.group(1)
                    index = summary_match.group(2)
                    if index not in models:
                        models[index] = {}
                    models[index]['summary'] = os.path.join(root, file_name)
                    if not base_name:
                        base_name = summary_base_name
                if full_data_match:
                    full_data_base_name = full_data_match.group(1)
                    index = full_data_match.group(2)
                    if index not in models:
                        models[index] = {}
                    models[index]['full_data'] = os.path.join(root, file_name)
                    if not base_name:
                        base_name = full_data_base_name

        return models, base_name
    except FileNotFoundError:
        st.error(f"Directory not found: {folder_path}")
        return None, None

# Streamlit interface for selecting directory and running analysis
st.set_page_config(layout="wide")  # Use full width of the screen
st.title("Protein Model Analysis")

# Provide options for how users want to upload files or specify a folder
file_option = st.radio("Choose how to provide files:", ("Upload Files", "Upload ZIP Folder"))

# Initialize variables
models = {}
uploaded_files = []
uploaded_zip = None
output_dir = ""
base_name = ""

# Display Sequence Information Button (works with all options)
if file_option == "Folder from Shared System":
    output_dir = st.text_input("Enter the path to your folder containing the CIF and JSON files")

    if st.button("Display Sequence Information"):
        models, base_name = scan_model_files(output_dir)
        if models:
            first_model = next(iter(models.values()))
            full_data_file = first_model.get('full_data')
            job_request_file = os.path.join(output_dir, f"{base_name}_job_request.json")
            if os.path.exists(job_request_file):
                display_sequence_info(job_request_file, full_data_file)
            else:
                st.error(f"Job request file {base_name}_job_request.json not found.")

elif file_option == "Upload Files":

    uploaded_files = st.file_uploader("Upload CIF and JSON files", type=['cif', 'json'], accept_multiple_files=True)

    if st.button("Display Sequence Information"):
        # Ensure necessary files are uploaded
        if not uploaded_files:
            st.error("Please upload the necessary CIF and JSON files.")
        else:
            job_request_file = next((file for file in uploaded_files if re.match(r'.*_job_request\.json', file.name)), None)
            full_data_file = next((file for file in uploaded_files if re.match(r'.*_full_data_\d+\.json', file.name)), None)
            if not (job_request_file and full_data_file):
                st.error("Missing required JSON files. Please upload all necessary files.")
            else:
                display_sequence_info(job_request_file, full_data_file)

elif file_option == "Upload ZIP Folder":
    uploaded_zip = st.file_uploader("Upload ZIP folder containing CIF and JSON files", type="zip")

    if st.button("Display Sequence Information"):
        if uploaded_zip:
            # Extract the uploaded ZIP file to a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save the uploaded ZIP file to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip_file:
                    tmp_zip_file.write(uploaded_zip.read())
                    tmp_zip_path = tmp_zip_file.name

                # Extract the ZIP file into the temporary directory
                with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Remove the temporary ZIP file
                os.remove(tmp_zip_path)

                models, base_name = scan_model_files(temp_dir)
                if models:
                    first_model = next(iter(models.values()))
                    full_data_file = first_model.get('full_data')
                    # Search for the job_request.json file with the correct naming
                    job_request_file = None
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if re.match(rf'{re.escape(base_name)}_job_request\.json', file):
                                job_request_file = os.path.join(root, file)
                                break
                        if job_request_file:
                            break

                    if job_request_file and full_data_file:
                        display_sequence_info(job_request_file, full_data_file)
                    else:
                        st.error(f"Required JSON files not found in the ZIP archive.")
                else:
                    st.error("No models found in the ZIP file.")

# ... [previous imports and functions remain unchanged] ...

# Analyze models button
if st.button("Analyze Models"):
    if file_option == "Folder from Shared System":
        models, base_name = scan_model_files(output_dir)

        if models:
            # Create a consolidated summary table for all models
            all_models_summary = []

            # Process models in sorted order
            for model_index in sorted(models.keys(), key=int):
                file_set = models[model_index]
                model_name = f"Model_{model_index}"
                summary_file = file_set.get('summary')

                if summary_file:
                    summary_data = extract_data(summary_file)
                    # Collect summary information for all models
                    model_summary_df = {
                        'Model Name': model_name,
                        'Fraction Disordered': summary_data.get('fraction_disordered', 'N/A'),
                        'Has Clash': summary_data.get('has_clash', 'N/A'),
                        'ipTM': summary_data.get('iptm', 'N/A'),
                        'pTM': summary_data.get('ptm', 'N/A'),
                        'Num Recycles': summary_data.get('num_recycles', 'N/A'),
                        'Ranking Score': summary_data.get('ranking_score', 'N/A')
                    }
                    all_models_summary.append(model_summary_df)
                else:
                    st.error(f"Missing summary file for {model_name}.")

            # Display consolidated summary before individual models
            if all_models_summary:
                consolidated_summary_df = pd.DataFrame(all_models_summary)
                st.write("### Consolidated Model Summary")
                st.dataframe(consolidated_summary_df)

            # Now process each model individually
            for model_index in sorted(models.keys(), key=int):
                file_set = models[model_index]
                model_name = f"Model_{model_index}"
                cif_file = file_set.get('model')
                full_data_file = file_set.get('full_data')
                summary_file = file_set.get('summary')

                if cif_file and full_data_file and summary_file:
                    full_data = extract_data(full_data_file)
                    summary_data = extract_data(summary_file)
                    if full_data and summary_data:
                        st.write(f"## {model_name}")
                        # Display 3D model and PAE heatmap side by side
                        col1, col2 = st.columns([1, 1], gap="medium")  # Equal column widths
                        with col1:
                            visualize_structure_with_molstar(cif_file)

                        with col2:
                            pae_matrix = full_data.get('pae', None)
                            if pae_matrix is not None:
                                plot_pae_heatmap(pae_matrix, model_name)

                        # Display ptm and ipTM averages and ipTM matrix
                        display_entity_ptm_iptm_averages_and_matrix(summary_data, model_name)
                    else:
                        st.error(f"Missing or invalid data for {model_name}.")
                else:
                    st.error(f"Missing required files for {model_name}.")

    elif file_option == "Upload Files":
        if uploaded_files:
            # Separate the files by model index
            model_files = {}
            base_name = None
            for file in uploaded_files:
                # Skip hidden files
                if file.name.startswith('.'):
                    continue

                # Extract base name and index
                model_match = re.match(r'(.*)_model_(\d+)\.cif', file.name)
                summary_match = re.match(r'(.*)_summary_confidences_(\d+)\.json', file.name)
                full_data_match = re.match(r'(.*)_full_data_(\d+)\.json', file.name)
                job_request_match = re.match(r'(.*)_job_request\.json', file.name)

                if job_request_match:
                    base_name = job_request_match.group(1)

                if model_match:
                    model_base_name = model_match.group(1)
                    index = model_match.group(2)
                    if index not in model_files:
                        model_files[index] = {}
                    model_files[index]['model'] = file
                    if not base_name:
                        base_name = model_base_name
                if summary_match:
                    summary_base_name = summary_match.group(1)
                    index = summary_match.group(2)
                    if index not in model_files:
                        model_files[index] = {}
                    model_files[index]['summary'] = file
                    if not base_name:
                        base_name = summary_base_name
                if full_data_match:
                    full_data_base_name = full_data_match.group(1)
                    index = full_data_match.group(2)
                    if index not in model_files:
                        model_files[index] = {}
                    model_files[index]['full_data'] = file
                    if not base_name:
                        base_name = full_data_base_name

            if model_files:
                # Create a consolidated summary table for all models
                all_models_summary = []

                # First, collect summary data for all models
                for model_index in sorted(model_files.keys(), key=int):
                    file_set = model_files[model_index]
                    model_name = f"Model_{model_index}"
                    summary_file = file_set.get('summary')

                    if summary_file:
                        summary_data = extract_data(summary_file)
                        # Collect summary information for all models
                        model_summary_df = {
                            'Model Name': model_name,
                            'Fraction Disordered': summary_data.get('fraction_disordered', 'N/A'),
                            'Has Clash': summary_data.get('has_clash', 'N/A'),
                            'ipTM': summary_data.get('iptm', 'N/A'),
                            'pTM': summary_data.get('ptm', 'N/A'),
                            'Num Recycles': summary_data.get('num_recycles', 'N/A'),
                            'Ranking Score': summary_data.get('ranking_score', 'N/A')
                        }
                        all_models_summary.append(model_summary_df)
                    else:
                        st.error(f"Missing summary file for {model_name}.")

                # Display consolidated summary before individual models
                if all_models_summary:
                    consolidated_summary_df = pd.DataFrame(all_models_summary)
                    st.write("### Consolidated Model Summary")
                    st.dataframe(consolidated_summary_df)

                # Now process each model individually
                for model_index in sorted(model_files.keys(), key=int):
                    file_set = model_files[model_index]
                    model_name = f"Model_{model_index}"
                    cif_file = file_set.get('model')
                    full_data_file = file_set.get('full_data')
                    summary_file = file_set.get('summary')

                    if cif_file and full_data_file and summary_file:
                        full_data = extract_data(full_data_file)
                        summary_data = extract_data(summary_file)

                        if full_data and summary_data:
                            st.write(f"## {model_name}")
                            # Display 3D model and PAE heatmap side by side
                            col1, col2 = st.columns([1, 1], gap="medium")
                            with col1:
                                visualize_structure_with_molstar(cif_file)

                            with col2:
                                pae_matrix = full_data.get('pae', None)
                                if pae_matrix is not None:
                                    plot_pae_heatmap(pae_matrix, model_name)

                            # Display ptm and ipTM averages and ipTM matrix
                            display_entity_ptm_iptm_averages_and_matrix(summary_data, model_name)
                        else:
                            st.error(f"Missing or invalid data for {model_name}.")
                    else:
                        st.error(f"Missing required files for {model_name}.")
            else:
                st.error("No valid model files found. Please check your uploads.")
        else:
            st.error("Please upload the necessary CIF and JSON files.")

    elif file_option == "Upload ZIP Folder":
        if uploaded_zip:
            # Extract the uploaded ZIP file to a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save the uploaded ZIP file to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip_file:
                    tmp_zip_file.write(uploaded_zip.read())
                    tmp_zip_path = tmp_zip_file.name

                # Extract the ZIP file into the temporary directory
                with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Remove the temporary ZIP file
                os.remove(tmp_zip_path)

                # Scan for model files in the extracted directory
                models, base_name = scan_model_files(temp_dir)

                if models:
                    # Create a consolidated summary table for all models
                    all_models_summary = []

                    # First, collect summary data for all models
                    for model_index in sorted(models.keys(), key=int):
                        file_set = models[model_index]
                        model_name = f"Model_{model_index}"
                        summary_file = file_set.get('summary')

                        if summary_file:
                            summary_data = extract_data(summary_file)
                            # Collect summary information for all models
                            model_summary_df = {
                                'Model Name': model_name,
                                'Fraction Disordered': summary_data.get('fraction_disordered', 'N/A'),
                                'Has Clash': summary_data.get('has_clash', 'N/A'),
                                'ipTM': summary_data.get('iptm', 'N/A'),
                                'pTM': summary_data.get('ptm', 'N/A'),
                                'Num Recycles': summary_data.get('num_recycles', 'N/A'),
                                'Ranking Score': summary_data.get('ranking_score', 'N/A')
                            }
                            all_models_summary.append(model_summary_df)
                        else:
                            st.error(f"Missing summary file for {model_name}.")

                    # Display consolidated summary before individual models
                    if all_models_summary:
                        consolidated_summary_df = pd.DataFrame(all_models_summary)
                        st.write("### Consolidated Model Summary")
                        st.dataframe(consolidated_summary_df)

                    # Now process each model individually
                    for model_index in sorted(models.keys(), key=int):
                        file_set = models[model_index]
                        model_name = f"Model_{model_index}"
                        cif_file = file_set.get('model')
                        full_data_file = file_set.get('full_data')
                        summary_file = file_set.get('summary')

                        if cif_file and full_data_file and summary_file:
                            full_data = extract_data(full_data_file)
                            summary_data = extract_data(summary_file)
                            if full_data and summary_data:
                                st.write(f"## {model_name}")
                                # Display 3D model and PAE heatmap side by side
                                col1, col2 = st.columns([1, 1], gap="medium")
                                with col1:
                                    visualize_structure_with_molstar(cif_file)

                                with col2:
                                    pae_matrix = full_data.get('pae', None)
                                    if pae_matrix is not None:
                                        plot_pae_heatmap(pae_matrix, model_name)

                                # Display ptm and ipTM averages and ipTM matrix
                                display_entity_ptm_iptm_averages_and_matrix(summary_data, model_name)
                        else:
                            st.error(f"Missing required files for {model_name}.")
                else:
                    st.error("No models found in the ZIP file.")

    st.write("Model analysis completed.")
