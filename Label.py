import streamlit as st
import pandas as pd
import os
import io
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT

# --- Page Configuration ---
st.set_page_config(
    page_title="Part Label Generator",
    page_icon="🏷️",
    layout="wide"
)

# --- Style Definitions ---
# Style for the two-part-per-label format
bold_style_v1 = ParagraphStyle(
    name='Bold_v1', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=14, spaceBefore=2, spaceAfter=2
)
# Style for the one-part-per-label format
bold_style_v2 = ParagraphStyle(
    name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=12, spaceBefore=0, spaceAfter=15,
)
desc_style = ParagraphStyle(
    name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16, spaceBefore=2, spaceAfter=2
)

# --- Formatting Functions ---
def format_part_no_v1(part_no): # For Multi-Part Label
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=17>{part1}</font><font size=22>{part2}</font></b>", bold_style_v1)
    return Paragraph(f"<b><font size=17>{part_no}</font></b>", bold_style_v1)

def format_part_no_v2(part_no): # For Single-Part Label
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if part_no.upper() == 'EMPTY':
         return Paragraph(f"<b><font size=34>EMPTY</font></b><br/><br/>", bold_style_v2)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b><br/><br/>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b><br/><br/>", bold_style_v2)

def format_description_v1(desc): # For Multi-Part Label
    if not desc or not isinstance(desc, str): desc = str(desc)
    font_size = 15 if len(desc) <= 30 else 13 if len(desc) <= 50 else 11 if len(desc) <= 70 else 10 if len(desc) <= 90 else 9
    desc_style_custom = ParagraphStyle(name='Description_v1', fontName='Helvetica', fontSize=font_size, alignment=TA_LEFT, leading=font_size + 2)
    return Paragraph(desc, desc_style_custom)

def format_description(desc): # For Single-Part Label
    if not desc or not isinstance(desc, str): desc = str(desc)
    return Paragraph(desc, desc_style)

# --- Core Logic Functions ---
def find_required_columns(df):
    cols = {col.upper().strip(): col for col in df.columns}
    part_no_key = next((k for k in cols if 'PART' in k and ('NO' in k or 'NUM' in k)), None)
    desc_key = next((k for k in cols if 'DESC' in k), None)
    bus_model_key = next((k for k in cols if 'BUS' in k and 'MODEL' in k), None)
    station_no_key = next((k for k in cols if 'STATION' in k), None)
    container_type_key = next((k for k in cols if 'CONTAINER' in k), None)
    # New: Find the Packaging Factor column
    packaging_factor_key = next((k for k in cols if 'PACKAGING' in k and 'FACTOR' in k), None)
    
    return (cols.get(part_no_key), cols.get(desc_key), cols.get(bus_model_key),
            cols.get(station_no_key), cols.get(container_type_key), cols.get(packaging_factor_key))

def parse_dimensions(dim_str):
    if not isinstance(dim_str, str) or not dim_str: return 0, 0
    nums = [int(n) for n in re.findall(r'\d+', dim_str)]
    return (nums[0], nums[1]) if len(nums) >= 2 else (0, 0)

def automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text=None):
    part_no_col, desc_col, model_col, station_col, container_col, packaging_factor_col = find_required_columns(df)
    
    if not all([part_no_col, station_col, packaging_factor_col]):
        st.error("❌ 'Part Number', 'Station No', or 'Packaging Factor' column not found.")
        return None

    df_processed = df.copy()
    # Use original column names for renaming
    rename_dict = {
        part_no_col: 'Part No', desc_col: 'Description',
        model_col: 'Bus Model', station_col: 'Station No', container_col: 'Container',
        packaging_factor_col: 'Packaging Factor'
    }
    df_processed.rename(columns={k: v for k, v in rename_dict.items() if k}, inplace=True)
    
    # --- NEW LOGIC: Create "Processing Units" based on Packaging Factor ---
    processing_units = []
    
    # Handle factor 0.5 parts (pairs)
    df_factor_half = df_processed[df_processed['Packaging Factor'] == 0.5].copy()
    for station, group in df_factor_half.groupby('Station No'):
        parts_list = group.to_dict('records')
        for i in range(0, len(parts_list), 2):
            if i + 1 < len(parts_list):
                pair = [parts_list[i], parts_list[i+1]]
                # Use the larger area of the two parts for sorting purposes
                area = max(parse_dimensions(bin_info_map.get(p['Container'], {}).get('dim_str', ''))[0] * parse_dimensions(bin_info_map.get(p['Container'], {}).get('dim_str', ''))[1] for p in pair)
                processing_units.append({'parts': pair, 'area': area, 'station': station})
            else: # Odd number of 0.5 parts in a station, treat the last one as a single
                processing_units.append({'parts': [parts_list[i]], 'area': 0, 'station': station})

    # Handle all other parts (factor 1 or other) as singles
    df_others = df_processed[df_processed['Packaging Factor'] != 0.5].copy()
    for part in df_others.to_dict('records'):
        area = parse_dimensions(bin_info_map.get(part['Container'], {}).get('dim_str', ''))[0] * parse_dimensions(bin_info_map.get(part['Container'], {}).get('dim_str', ''))[1]
        processing_units.append({'parts': [part], 'area': area, 'station': part['Station No']})
        
    # Sort all units by station, then by largest area first
    processing_units.sort(key=lambda x: (x['station'], -x['area']))

    # --- Placement Logic (works on units instead of individual parts) ---
    final_df_parts = []
    available_cells = []
    for rack_name, config in sorted(rack_configs.items()):
        rack_num_val = ''.join(filter(str.isdigit, rack_name))
        rack_num_1st, rack_num_2nd = (rack_num_val[0], rack_num_val[1]) if len(rack_num_val) > 1 else ('0', rack_num_val[0])
        for level in sorted(config.get('levels', [])):
            for i in range(config.get('cells_per_level', 0)):
                location = {'Level': level, 'Physical_Cell': f"{i+1:02d}", 'Rack': base_rack_id, 'Rack No 1st': rack_num_1st, 'Rack No 2nd': rack_num_2nd}
                available_cells.append(location)

    current_cell_index = 0
    for unit in processing_units:
        if current_cell_index >= len(available_cells):
            st.error(f"❌ Ran out of rack space. Could not place {len(unit['parts'])} part(s).")
            break
        
        current_location = available_cells[current_cell_index]
        for part in unit['parts']:
            part.update(current_location)
            final_df_parts.append(part)
        current_cell_index += 1

    for i in range(current_cell_index, len(available_cells)):
        empty_part = {'Part No': 'EMPTY', 'Description': '', 'Bus Model': '', 'Station No': 'N/A', 'Container': ''}
        empty_part.update(available_cells[i])
        final_df_parts.append(empty_part)

    return pd.DataFrame(final_df_parts) if final_df_parts else pd.DataFrame()


def assign_sequential_location_ids(df):
    df_sorted = df.sort_values(by=['Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']).copy()
    df_parts_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    
    location_counters = {}
    sequential_ids = []
    for index, row in df_parts_only.iterrows():
        rack_id = (row['Rack No 1st'], row['Rack No 2nd'])
        level = row['Level']
        counter_key = (rack_id, level)
        if counter_key not in location_counters:
            location_counters[counter_key] = 1
        
        current_id_num = location_counters[counter_key]
        sequential_ids.append(current_id_num)
        location_counters[counter_key] += 1
        
    df_parts_only['Cell'] = sequential_ids
    
    df_empty_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() == 'EMPTY'].copy()
    df_empty_only['Cell'] = df_empty_only['Physical_Cell']

    return pd.concat([df_parts_only, df_empty_only], ignore_index=True)

def extract_location_values(row, cell_override=None):
    cell_val = cell_override if cell_override is not None else row.get('Cell', '')
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level']] + [str(cell_val)]

# --- Combined PDF Generation ---
def generate_combined_labels(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    
    df.sort_values(by=['Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell'], inplace=True)
    df_parts_only = df[df['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    
    physical_cell_grouping_cols = ['Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']
    df_grouped = df_parts_only.groupby(physical_cell_grouping_cols)
    
    total_labels, label_count, label_summary = len(df_grouped), 0, {}

    for i, (group_key, group) in enumerate(df_grouped):
        if progress_bar: progress_bar.progress(int((i / total_labels) * 100))
        if status_text: status_text.text(f"Processing Label {i+1}/{total_labels}")

        if label_count > 0 and label_count % 4 == 0: elements.append(PageBreak())

        part1 = group.iloc[0].to_dict()
        
        # --- Automatic Selection Logic ---
        if group.iloc[0]['Packaging Factor'] == 0.5 and len(group) > 1:
            # MULTIPLE PARTS (Factor 0.5)
            part2 = group.iloc[1].to_dict()
            part_table1 = Table([['Part No', format_part_no_v1(part1['Part No'])], ['Description', format_description_v1(part1['Description'])]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
            part_table2 = Table([['Part No', format_part_no_v1(part2['Part No'])], ['Description', format_description_v1(part2['Description'])]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
            
            location_values = extract_location_values(part1, cell_override=part1['Cell']) # Use first part's ID
            location_data = [['Line Location'] + location_values]
            col_proportions = [1.8, 2.7, 1.3, 1.3, 1.3, 1.3, 1.3]
            location_widths = [4*cm] + [w * (11*cm) / sum(col_proportions) for w in col_proportions]
            location_table = Table(location_data, colWidths=location_widths, rowHeights=0.8*cm)
            
            elements.extend([part_table1, Spacer(1, 0.3*cm), part_table2, Spacer(1, 0.3*cm), location_table])

        else:
            # SINGLE PART (Factor 1 or other)
            part_table = Table([['Part No', format_part_no_v2(part1['Part No'])], ['Description', format_description(part1['Description'])]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
            location_values = extract_location_values(part1)
            location_data = [['Line Location'] + location_values]
            col_widths = [1.7, 2.9, 1.3, 1.2, 1.3, 1.3, 1.3]
            location_widths = [4*cm] + [w * (11*cm) / sum(col_widths) for w in col_widths]
            location_table = Table(location_data, colWidths=location_widths, rowHeights=0.9*cm)
            
            elements.extend([part_table, Spacer(1, 0.3*cm), location_table])

        rack_num = f"{part1.get('Rack No 1st', '0')}{part1.get('Rack No 2nd', '0')}"
        rack_key = f"Rack {rack_num.zfill(2)}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
        elements.append(Spacer(1, 0.2*cm))
        label_count += 1
        
    if elements: doc.build(elements)
    buffer.seek(0)
    return buffer, label_summary

# --- Main Application UI ---
def main():
    st.title("🏷️ Rack Label Generator")
    st.markdown("<p style='font-style:italic;'>Designed by Agilomatrix</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.sidebar.title("📄 Label Options")
    base_rack_id = st.sidebar.text_input("Enter Storage Line Side Infrastructure", "R")
    
    uploaded_file = st.file_uploader("Choose an Excel or CSV file with a 'Packaging Factor' column", type=['xlsx', 'xls', 'csv'])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, keep_default_na=False) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file, keep_default_na=False)
            st.success(f"✅ File loaded! Found {len(df)} rows.")
            
            part_no_col, desc_col, bus_model_col, station_col, container_col, packaging_factor_col = find_required_columns(df)
            
            if not packaging_factor_col:
                st.error("❌ Critical Error: A column containing 'Packaging Factor' was not found in your file. This column is required.")
            else:
                st.sidebar.markdown("---")
                st.sidebar.subheader("Global Rack Configuration")
                levels = st.multiselect("Active Levels (for all racks)", options=['A','B','C','D','E','F','G','H'], default=['A','B','C','D'])
                num_cells_per_level = st.sidebar.number_input("Number of Physical Cells per Level", min_value=1, value=10, step=1)
                num_racks = st.sidebar.number_input("Total Number of Racks", min_value=1, value=4, step=1)
                
                # Get original column name for container type
                _, _, _, _, container_col_orig, _ = find_required_columns(df)
                unique_containers = get_unique_containers(df, container_col_orig)
                bin_info_map = {}
                st.sidebar.markdown("---")
                st.sidebar.subheader("Container (Bin) Dimensions")
                for container in unique_containers:
                    dim = st.sidebar.text_input(f"Dimensions for {container}", key=f"bindim_{container}", placeholder="e.g., 600x400")
                    # We no longer need capacity here, but keep dimensions for sorting
                    bin_info_map[container] = {'dim_str': dim}

                if st.button("🚀 Generate PDF Labels", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    try:
                        rack_configs = {}
                        for i in range(num_racks):
                            rack_name = f"Rack {i+1:02d}"
                            rack_configs[rack_name] = {'levels': levels, 'cells_per_level': num_cells_per_level}

                        df_physically_assigned = automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text)
                        df_final_labels = assign_sequential_location_ids(df_physically_assigned)
                        
                        if df_final_labels is not None and not df_final_labels.empty:
                            pdf_buffer, label_summary = generate_combined_labels(df_final_labels, progress_bar, status_text)
                            
                            if pdf_buffer:
                                total_labels = sum(label_summary.values())
                                status_text.text(f"✅ PDF with {total_labels} labels generated successfully!")
                                file_name = f"{os.path.splitext(uploaded_file.name)[0]}_labels.pdf"
                                st.download_button(label="📥 Download PDF", data=pdf_buffer.getvalue(), file_name=file_name, mime="application/pdf")

                                if total_labels > 0:
                                    st.markdown("---")
                                    st.subheader("📊 Generation Summary")
                                    st.markdown(f"A total of **{total_labels}** labels (one per physical cell) have been generated.")
                                    summary_df = pd.DataFrame(list(label_summary.items()), columns=['Rack', 'Number of Labels'])
                                    st.table(summary_df.sort_values(by='Rack').reset_index(drop=True))
                        else:
                            st.error("❌ No data was processed. Check your input file.")
                    except Exception as e:
                        st.error(f"❌ An unexpected error occurred: {e}")
                        st.exception(e)
                    finally:
                        progress_bar.empty()
                        status_text.empty()

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    else:
        st.info("👆 Upload a file to begin.")

if __name__ == "__main__":
    main()
