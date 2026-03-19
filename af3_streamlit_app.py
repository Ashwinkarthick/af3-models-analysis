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
import shlex
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

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

# Helper function to extract PAE matrix from AF3 or AF2 style JSON
def extract_pae_matrix(json_data):
    if json_data is None:
        return None

    if isinstance(json_data, dict):
        if 'pae' in json_data:
            return np.array(json_data['pae'])
        if 'predicted_aligned_error' in json_data:
            return np.array(json_data['predicted_aligned_error'])
        if all(key in json_data for key in ('residue1', 'residue2', 'distance')):
            residue1 = np.array(json_data['residue1'], dtype=int)
            residue2 = np.array(json_data['residue2'], dtype=int)
            distance = np.array(json_data['distance'], dtype=float)
            size = max(residue1.max(), residue2.max())
            matrix = np.full((size, size), np.nan)
            matrix[residue1 - 1, residue2 - 1] = distance
            return matrix

    if isinstance(json_data, list) and len(json_data) > 0 and isinstance(json_data[0], dict):
        first_item = json_data[0]
        if 'predicted_aligned_error' in first_item:
            return np.array(first_item['predicted_aligned_error'])
        if all(key in first_item for key in ('residue1', 'residue2', 'distance')):
            residue1 = np.array(first_item['residue1'], dtype=int)
            residue2 = np.array(first_item['residue2'], dtype=int)
            distance = np.array(first_item['distance'], dtype=float)
            size = max(residue1.max(), residue2.max())
            matrix = np.full((size, size), np.nan)
            matrix[residue1 - 1, residue2 - 1] = distance
            return matrix

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

# Function to plot PAE heatmap and capture interactive residue selection
def plot_pae_heatmap(pae_matrix, model_name, plot_key, selected_pair=None):
    num_residues = len(pae_matrix)
    cell_size = 20
    fig_height = min(num_residues * cell_size, 800)

    fig = go.Figure(
        data=go.Heatmap(
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
            zmin=np.nanmin(pae_matrix),
            zmax=np.nanmax(pae_matrix),
            showscale=True,
        )
    )

    if selected_pair:
        x_idx, y_idx = selected_pair
        fig.add_shape(
            type='rect',
            x0=x_idx - 0.5,
            y0=y_idx - 0.5,
            x1=x_idx + 0.5,
            y1=y_idx + 0.5,
            line=dict(color='red', width=2),
        )

    fig.update_xaxes(tickmode='linear', tick0=0, dtick=max(1, num_residues // 8), showgrid=False)
    fig.update_yaxes(
        tickmode='linear',
        tick0=0,
        dtick=max(1, num_residues // 8),
        showgrid=False,
        scaleanchor='x',
        scaleratio=1,
    )
    fig.update_layout(
        title=f"{model_name} PAE Heatmap",
        xaxis_title='Scored Residue',
        yaxis_title='Aligned Residue',
        autosize=False,
        height=fig_height,
        margin=dict(l=50, r=50, t=50, b=50),
        clickmode='event+select',
        dragmode='select',
    )

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        key=plot_key,
        on_select='rerun',
        selection_mode='points',
    )

    try:
        points = event.get('selection', {}).get('points', []) if isinstance(event, dict) else []
        if points:
            point = points[-1]
            x = int(round(point.get('x')))
            y = int(round(point.get('y')))
            return (x, y)
    except Exception:
        pass

    return selected_pair


# Function to parse residue mapping from CIF text
def build_cif_residue_index(cif_text):
    lines = cif_text.splitlines()
    headers = []
    data_lines = []
    in_atom_loop = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == 'loop_':
            local_headers = []
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith('_atom_site.'):
                local_headers.append(lines[j].strip())
                j += 1
            if local_headers:
                headers = local_headers
                k = j
                while k < len(lines):
                    row = lines[k].strip()
                    if not row or row.startswith('#') or row.startswith('loop_') or row.startswith('_'):
                        break
                    data_lines.append(row)
                    k += 1
                in_atom_loop = True
                break

    if not in_atom_loop or not headers:
        return []

    header_index = {h: idx for idx, h in enumerate(headers)}
    chain_col = header_index.get('_atom_site.auth_asym_id', header_index.get('_atom_site.label_asym_id'))
    seq_col = header_index.get('_atom_site.auth_seq_id', header_index.get('_atom_site.label_seq_id'))

    if chain_col is None or seq_col is None:
        return []

    residue_order = []
    seen = set()
    for row in data_lines:
        try:
            tokens = shlex.split(row)
        except ValueError:
            tokens = row.split()
        if len(tokens) <= max(chain_col, seq_col):
            continue
        chain_id = tokens[chain_col]
        seq_id = tokens[seq_col]
        key = (chain_id, seq_id)
        if key not in seen:
            seen.add(key)
            residue_order.append(key)

    return residue_order


# Function to create a 3D visualization using CIF content and optional highlighted residues
def visualize_structure_with_molstar(cif_file_or_path, selected_pair=None, viewer_key='viewer'):
    if isinstance(cif_file_or_path, str):
        with open(cif_file_or_path, 'r', encoding='utf-8') as f:
            cif_text = f.read()
    else:
        cif_text = cif_file_or_path.getvalue().decode('utf-8')

    residue_index = build_cif_residue_index(cif_text)
    selected_residues = []
    if selected_pair and residue_index:
        for idx in [selected_pair[0] + 1, selected_pair[1] + 1]:
            if 1 <= idx <= len(residue_index):
                selected_residues.append(residue_index[idx - 1])

    selection_clause = ' or '.join([f"(:{chain} and {resi})" for chain, resi in selected_residues]) if selected_residues else ''

    html = f"""
    <div id="ngl-viewer-{viewer_key}" style="width: 100%; height: 600px;"></div>
    <script src="https://unpkg.com/ngl@2.0.0-dev.37/dist/ngl.js"></script>
    <script>
      const stage = new NGL.Stage('ngl-viewer-{viewer_key}', {{ backgroundColor: 'white' }});
      const blob = new Blob([{json.dumps(cif_text)}], {{ type: 'text/plain' }});
      stage.loadFile(blob, {{ ext: 'cif' }}).then(function (comp) {{
        comp.addRepresentation('cartoon', {{ color: 'chainid' }});
        const sele = {json.dumps(selection_clause)};
        if (sele) {{
          comp.addRepresentation('ball+stick', {{ sele: sele, color: 'red' }});
          comp.autoView(sele);
        }} else {{
          comp.autoView();
        }}
      }});
      window.addEventListener('resize', function() {{ stage.handleResize(); }});
    </script>
    """

    components.html(html, height=620, key=f"ngl_{viewer_key}_{selection_clause}")
# Function to create a 3D visualization using Mol*
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
                af2_model_match = re.match(r'ranked_(\d+)\.cif', file_name)
                af2_pae_match = re.match(r'pae_model_\d+_.*_pred_(\d+)\.json', file_name)
                af2_pae_ranked_match = re.match(r'pae_ranked_(\d+)\.json', file_name)

                if model_match:
                    model_base_name = model_match.group(1)
                    index = model_match.group(2)
                    if index not in models:
                        models[index] = {}
                    models[index]['model'] = os.path.join(root, file_name)
                    models[index]['format'] = 'af3'
                    if not base_name:
                        base_name = model_base_name
                if summary_match:
                    summary_base_name = summary_match.group(1)
                    index = summary_match.group(2)
                    if index not in models:
                        models[index] = {}
                    models[index]['summary'] = os.path.join(root, file_name)
                    models[index]['format'] = 'af3'
                    if not base_name:
                        base_name = summary_base_name
                if full_data_match:
                    full_data_base_name = full_data_match.group(1)
                    index = full_data_match.group(2)
                    if index not in models:
                        models[index] = {}
                    models[index]['full_data'] = os.path.join(root, file_name)
                    models[index]['format'] = 'af3'
                    if not base_name:
                        base_name = full_data_base_name
                if af2_model_match:
                    index = af2_model_match.group(1)
                    if index not in models:
                        models[index] = {}
                    models[index]['model'] = os.path.join(root, file_name)
                    models[index].setdefault('format', 'af2')
                if af2_pae_match:
                    index = af2_pae_match.group(1)
                    if index not in models:
                        models[index] = {}
                    models[index]['full_data'] = os.path.join(root, file_name)
                    models[index].setdefault('format', 'af2')
                if af2_pae_ranked_match:
                    index = af2_pae_ranked_match.group(1)
                    if index not in models:
                        models[index] = {}
                    models[index]['full_data'] = os.path.join(root, file_name)
                    models[index].setdefault('format', 'af2')

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
                    summary_data = extract_data(summary_file) if summary_file else None
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
                elif file_set.get('format', 'af3') == 'af3':
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

                if cif_file and full_data_file:
                    full_data = extract_data(full_data_file)
                    summary_data = extract_data(summary_file) if summary_file else None
                    if full_data:
                        st.write(f"## {model_name}")
                        # Display 3D model and PAE heatmap side by side with linked selection
                        pae_matrix = extract_pae_matrix(full_data)
                        if pae_matrix is not None:
                            selection_state_key = f"selected_pair_{model_name}"
                            plot_state_key = f"pae_plot_{model_name}"
                            selected_pair = st.session_state.get(selection_state_key)

                            col1, col2 = st.columns([1, 1], gap="medium")  # Equal column widths
                            with col2:
                                selected_pair = plot_pae_heatmap(pae_matrix, model_name, plot_state_key, selected_pair)
                                st.session_state[selection_state_key] = selected_pair
                                if selected_pair:
                                    st.caption(f"Selected PAE cell: aligned residue {selected_pair[1] + 1}, scored residue {selected_pair[0] + 1}")

                            with col1:
                                visualize_structure_with_molstar(cif_file, selected_pair=selected_pair, viewer_key=model_name)

                        # Display ptm and ipTM averages and ipTM matrix (AF3 summary files)
                        if summary_data:
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
                af2_model_match = re.match(r'ranked_(\d+)\.cif', file.name)
                af2_pae_match = re.match(r'pae_model_\d+_.*_pred_(\d+)\.json', file.name)
                af2_pae_ranked_match = re.match(r'pae_ranked_(\d+)\.json', file.name)
                job_request_match = re.match(r'(.*)_job_request\.json', file.name)

                if job_request_match:
                    base_name = job_request_match.group(1)

                if model_match:
                    model_base_name = model_match.group(1)
                    index = model_match.group(2)
                    if index not in model_files:
                        model_files[index] = {}
                    model_files[index]['model'] = file
                    model_files[index]['format'] = 'af3'
                    if not base_name:
                        base_name = model_base_name
                if summary_match:
                    summary_base_name = summary_match.group(1)
                    index = summary_match.group(2)
                    if index not in model_files:
                        model_files[index] = {}
                    model_files[index]['summary'] = file
                    model_files[index]['format'] = 'af3'
                    if not base_name:
                        base_name = summary_base_name
                if full_data_match:
                    full_data_base_name = full_data_match.group(1)
                    index = full_data_match.group(2)
                    if index not in model_files:
                        model_files[index] = {}
                    model_files[index]['full_data'] = file
                    model_files[index]['format'] = 'af3'
                    if not base_name:
                        base_name = full_data_base_name
                if af2_model_match:
                    index = af2_model_match.group(1)
                    if index not in model_files:
                        model_files[index] = {}
                    model_files[index]['model'] = file
                    model_files[index].setdefault('format', 'af2')
                if af2_pae_match:
                    index = af2_pae_match.group(1)
                    if index not in model_files:
                        model_files[index] = {}
                    model_files[index]['full_data'] = file
                    model_files[index].setdefault('format', 'af2')
                if af2_pae_ranked_match:
                    index = af2_pae_ranked_match.group(1)
                    if index not in model_files:
                        model_files[index] = {}
                    model_files[index]['full_data'] = file
                    model_files[index].setdefault('format', 'af2')

            if model_files:
                # Create a consolidated summary table for all models
                all_models_summary = []

                # First, collect summary data for all models
                for model_index in sorted(model_files.keys(), key=int):
                    file_set = model_files[model_index]
                    model_name = f"Model_{model_index}"
                    summary_file = file_set.get('summary')

                    if summary_file:
                        summary_data = extract_data(summary_file) if summary_file else None
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
                    elif file_set.get('format', 'af3') == 'af3':
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

                    if cif_file and full_data_file:
                        full_data = extract_data(full_data_file)
                        summary_data = extract_data(summary_file) if summary_file else None

                        if full_data:
                            st.write(f"## {model_name}")
                            # Display 3D model and PAE heatmap side by side with linked selection
                            pae_matrix = extract_pae_matrix(full_data)
                            if pae_matrix is not None:
                                selection_state_key = f"selected_pair_{model_name}"
                                plot_state_key = f"pae_plot_{model_name}"
                                selected_pair = st.session_state.get(selection_state_key)

                                col1, col2 = st.columns([1, 1], gap="medium")
                                with col2:
                                    selected_pair = plot_pae_heatmap(pae_matrix, model_name, plot_state_key, selected_pair)
                                    st.session_state[selection_state_key] = selected_pair
                                    if selected_pair:
                                        st.caption(f"Selected PAE cell: aligned residue {selected_pair[1] + 1}, scored residue {selected_pair[0] + 1}")

                                with col1:
                                    visualize_structure_with_molstar(cif_file, selected_pair=selected_pair, viewer_key=model_name)

                            # Display ptm and ipTM averages and ipTM matrix (AF3 summary files)
                            if summary_data:
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
                            summary_data = extract_data(summary_file) if summary_file else None
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
                        elif file_set.get('format', 'af3') == 'af3':
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

                        if cif_file and full_data_file:
                            full_data = extract_data(full_data_file)
                            summary_data = extract_data(summary_file) if summary_file else None
                            if full_data:
                                st.write(f"## {model_name}")
                                # Display 3D model and PAE heatmap side by side with linked selection
                                pae_matrix = extract_pae_matrix(full_data)
                                if pae_matrix is not None:
                                    selection_state_key = f"selected_pair_{model_name}"
                                    plot_state_key = f"pae_plot_{model_name}"
                                    selected_pair = st.session_state.get(selection_state_key)

                                    col1, col2 = st.columns([1, 1], gap="medium")
                                    with col2:
                                        selected_pair = plot_pae_heatmap(pae_matrix, model_name, plot_state_key, selected_pair)
                                        st.session_state[selection_state_key] = selected_pair
                                        if selected_pair:
                                            st.caption(f"Selected PAE cell: aligned residue {selected_pair[1] + 1}, scored residue {selected_pair[0] + 1}")

                                    with col1:
                                        visualize_structure_with_molstar(cif_file, selected_pair=selected_pair, viewer_key=model_name)

                                # Display ptm and ipTM averages and ipTM matrix (AF3 summary files)
                                if summary_data:
                                    display_entity_ptm_iptm_averages_and_matrix(summary_data, model_name)
                        else:
                            st.error(f"Missing required files for {model_name}.")
                else:
                    st.error("No models found in the ZIP file.")

    st.write("Model analysis completed.")
