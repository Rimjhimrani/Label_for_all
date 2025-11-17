import streamlit as st
import pandas as pd
import os
import io
import re
from reportlab.lib.pagesizes_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate,=None):
    # This function uses the original column names found by find_required_columns
    part_no_col, desc_col, model_col, station_col, container_col, packaging_factor_col Table, TableStyle, Spacer, Paragraph, PageBreak
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT

# --- Page Configuration = find_required_columns(df)
    
    if not all([part_no_col, station_col, packaging_factor_col]):
        st.error("❌ 'Part Number', 'Station No', or ---
st.set_page_config(
    page_title="Part Label Generator",
    page_icon="🏷 'Packaging Factor' column not found.")
        return None

    # Create a copy to avoid modifying the original DataFrame️",
    layout="wide"
)

# --- Style Definitions ---
bold_style_v1 = Paragraph
    df_processed = df.copy()

    # Rename columns to a standard format for easier processing
    rename_Style(
    name='Bold_v1', fontName='Helvetica-Bold', fontSize=10, alignmentdict = {
        part_no_col: 'Part No', desc_col: 'Description', model_col=TA_LEFT, leading=14, spaceBefore=2, spaceAfter=2
)
bold_: 'Bus Model', 
        station_col: 'Station No', container_col: 'Container', packagingstyle_v2 = ParagraphStyle(
    name='Bold_v2', fontName='Helvetica-Bold',_factor_col: 'Packaging Factor'
    }
    # Only rename columns that were actually found
     fontSize=10, alignment=TA_LEFT, leading=12, spaceBefore=0, spaceAfter=1df_processed.rename(columns={k: v for k, v in rename_dict.items() if k},5,
)
desc_style = ParagraphStyle(
    name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16, spaceBefore=2, spaceAfter=2
)

# --- Formatting Functions ---
def format_part_no_v1(part_no):
    if not inplace=True)

    processing_units = []
    
    df_factor_half = df_processed[df_processed['Packaging Factor'] == 0.5].copy()
    for station, group in df_factor part_no or not isinstance(part_no, str): part_no = str(part_no)
    if_half.groupby('Station No'):
        parts_list = group.to_dict('records')
         len(part_no) > 5:
        part1, part2 = part_no[:-5],for i in range(0, len(parts_list), 2):
            if i + 1 < len part_no[-5:]
        return Paragraph(f"<b><font size=17>{part1}</(parts_list):
                pair = [parts_list[i], parts_list[i+1]]
                font><font size=22>{part2}</font></b>", bold_style_v1)
    area = max(parse_dimensions(bin_info_map.get(p.get('Container', ''), {}return Paragraph(f"<b><font size=17>{part_no}</font></b>", bold_style_v1).get('dim_str', ''))[0] * parse_dimensions(bin_info_map.get(p.get('Container', ''), {}).get('dim_str', ''))[1] for p in)

def format_part_no_v2(part_no):
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if part_no. pair)
                processing_units.append({'parts': pair, 'area': area, 'station': station})
            elseupper() == 'EMPTY':
         return Paragraph(f"<b><font size=34>EMPTY</font></b><br/><br/>", bold_style_v2)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b><br/><br/>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b><br/><br/>", bold_style_v2)

def format_description_v1(desc):
    if not desc or not isinstance(desc, str): desc = str(desc)
    font_size = 15 if len(desc) <= 30 else 13 if len(desc) <= 50 else 11 if len(desc) <= 70 else 10 if len(desc) <= 90 else 9
    desc_style_custom = ParagraphStyle(name='Description_v1', fontName='Helvetica', fontSize=font_size, alignment=TA_LEFT, leading=font_size + 2)
    return Paragraph(desc, desc_style_custom)

def format_description(desc):
    if not desc or not isinstance(desc, str): desc = str(desc)
    return Paragraph(desc, desc_style)

# --- Core Logic Functions ---
def find_required_columns(df):
    # Create a mapping of uppercase/stripped column names to their original names
    cols_map = {col.upper().strip(): col for col in df.columns}
    # Get a list of the uppercase/stripped names for searching
    upper_cols = list(cols_map.keys())
    
    # Find the keys for the required columns
    part_no_key = next((k for k in upper_cols if 'PART' in k and ('NO' in k or 'NUM' in k)), None)
    desc_key = next((k for k in upper_cols if 'DESC' in k), None)
    bus_model_key = next((k for k in upper_cols if 'BUS' in k and 'MODEL' in k), None)
    station_no_key = next((k for k in upper_cols if 'STATION' in k), None)
    container_type_key = next((k for k in upper_cols if 'CONTAINER' in k), None)
    packaging_factor_key = next((k for k in upper_cols if 'PACKAGING' in k and 'FACTOR' in k), None)
    
    # Return the original column names using the map
    return (cols_map.get(part_no_key), cols_map.get(desc_key), cols_map.get(bus_model_key),
            cols_map.get(station_no_key), cols_map.get(container_type_key), cols_map.get(packaging_factor_key))

# --- THIS IS THE RESTORED FUNCTION ---
def get_unique_containers(df, container_col):
    if not container_col or container_col not in df.columns: return []
    return sorted(df[container_col].dropna().astype(str).unique())

def parse_dimensions(dim_str):
    if not isinstance(dim:
                processing_units.append({'parts': [parts_list[i]], 'area': 0, 'station': station})

    df_others = df_processed[df_processed['Packaging Factor'] != 0.5].copy()
    for part in df_others.to_dict('records'):
        area = parse_dimensions(bin_info_map.get(part.get('Container', ''), {}).get('dim_str', ''))[0] * parse_dimensions(bin_info_map.get(part.get('Container', ''), {}).get('dim_str', ''))[1]
        processing_units.append({'parts': [part], 'area': area, 'station': part.get('Station No', 'N/A')})
        
    processing_units.sort(key=lambda x: (x['station'], -x['area']))

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
    for index, row in df_parts_only.iterrows_str, str) or not dim_str: return 0, 0
    nums = [int(n) for n in re.findall(r'\d+', dim_str)]
    return (nums[0], nums[1]) if len(nums) >= 2 else (0, 0)

def automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text=None):
    part_no_col, desc_col, model_col, station_col, container_col, packaging_factor_col = find_required_columns(df)
    
    if not all([part_no_col, station_col, packaging_factor_col]):
        st.error("❌ 'Part Number', 'Station No', or 'Packaging Factor' column not found.")
        return None

    df_processed = df.copy()
    rename_dict = {
        part_no_col: 'Part No', desc_col: 'Description',
        model_col: 'Bus Model', station_col: 'Station No', container_col: 'Container',
        packaging_factor_col: 'Packaging Factor'
    }
    df_processed.rename(columns={k: v for k, v in rename_dict.items() if k}, inplace=True)
    
    processing_units = []
    
    df_factor_half = df_processed[df_processed['Packaging Factor'] == 0.5].():
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

def generate_combined_labels(df, progress_bar=None, status_text=Nonecopy()
    for station, group in df_factor_half.groupby('Station No'):
        parts_list = group.to_dict('records')
        for i in range(0, len(parts_list), 2):
            if i + 1 < len(parts_list):
                pair = [parts_list[i], parts_list[i+1]]
                area = max(parse_dimensions(bin_info_map.get(p['Container'], {}).get('dim_str', ''))[0] * parse_dimensions(bin_info_map.get(p['Container'], {}).get('dim_str', ''))[1] for p in pair)
                processing_units.append({'parts': pair, 'area': area, 'station':):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    
    df.sort_values(by=['Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell'], inplace=True)
    df_parts_only = df[df['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    
    physical_cell_grouping_cols = ['Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']
    df_grouped = df_parts_only.groupby station})
            else:
                processing_units.append({'parts': [parts_list[i]], 'area': 0, 'station': station})

    df_others = df_processed[df_processed['Packaging Factor'] != 0.5].copy()
    for part in df_others.to_dict('records'):
        area = parse_dimensions(bin_info_map.get(part['Container'], {}).get('dim_str', ''))[0] * parse_dimensions(bin_info_map.get(part['Container'], {}).get('dim_str', ''))[1]
        processing_units.append({'parts': [part], '(physical_cell_grouping_cols)
    
    total_labels, label_count, label_summary = len(df_grouped), 0, {}

    for i, (group_key, group) in enumerate(df_grouped):
        if progress_bar: progress_bar.progress(int((i / total_labels) * 100))
        if status_text: status_text.text(f"Processing Label {i+1}/{total_labels}")

        if label_count > 0 and label_count % 4 == 0: elements.append(PageBreak())

        part1 = group.iloc[0].to_dict()
        
        if part1.get('Packaging Factor') == 0.5 and len(group) > 1:
area': area, 'station': part['Station No']})
        
    processing_units.sort(key=lambda x: (x['station'], -x['area']))

    final_df_parts = []
    available_cells = []
    for rack_name, config in sorted(rack_configs.items()):
        rack_num_val = ''.join(filter(str.isdigit, rack_name))
        rack_num_1            part2 = group.iloc[1].to_dict()
            part_table1 = Table([['Part No', format_part_no_v1(part1['Part No'])], ['Description', format_description_v1(part1.get('Description', ''))]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
            part_table2 = Table([['Part No', format_part_no_v1(part2['Part No'])], ['Description', format_description_v1(part2.get('Description', ''))]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
            
            location_valuesst, rack_num_2nd = (rack_num_val[0], rack_num_val[1]) if len(rack_num_val) > 1 else ('0', rack_num_val[0])
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
        empty_part = {'Part No': 'EMPTY', 'Description': '', 'Bus Model': '', 'Station No': ' = extract_location_values(part1, cell_override=part1['Cell'])
            location_data = [['Line Location'] + location_values]
            col_proportions = [1.8, 2.7, 1.3, 1.3, 1.3, 1.3, 1.3]
            location_widths = [4*cm] + [w * (11*cm) / sum(col_proportions) for w in col_proportions]
            location_table = Table(location_data, colWidths=location_widths, rowHeights=0.8*cm)
            
            elements.N/A', 'Container': ''}
        empty_part.update(available_cells[i])
        final_df_parts.append(empty_part)

    return pd.DataFrame(final_df_parts) if final_df_parts else pd.DataFrame()


def assign_sequential_location_ids(df):
    df_sorted = df.sort_values(by=['Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']).copy()
    df_parts_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    
    location_counters, sequential_ids = {}, []
    for index, row in df_parts_only.iterrows():
        rack_id = (rowextend([part_table1, Spacer(1, 0.3*cm), part_table2, Spacer(1, 0.3*cm), location_table])

        else:
            part_table = Table([['Part No', format_part_no_v2(part1['Part No'])], ['Description', format_description(part1.get('Description', ''))]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
            location_values = extract_location_values(part1)
            location_data = [['Line Location'] + location_values]
            col_widths = [1.7, 2.9, 1.3, 1.2, 1.3, 1.3, 1.3]
            location_widths = [4*cm] + [w * (11*cm) / sum(col_widths) for w in col_widths]
            location_table = Table(location_data, colWidths=location_widths['Rack No 1st'], row['Rack No 2nd'])
        level = row['Level']
        counter_key = (rack_id, level)
        if counter_key not in location_counters:
            location_counters[counter_key] = 1
        
        current_id_num = location, rowHeights=0.9*cm)
            
            elements.extend([part_table, Spacer(1, 0.3*cm), location_table])

        rack_num = f"{part1.get('Rack_counters[counter_key]
        sequential_ids.append(current_id_num)
        location_counters[counter_key] += 1
        
    df_parts_only['Cell'] = sequential_ids
    
    df_empty_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() == 'EMPTY'].copy()
    df_empty_only['Cell'] = df_empty_only['Physical_Cell']

    return pd.concat([df_parts_only, df_empty_only], ignore_index=True)

def extract_location_values(row, cell_override=None):
    cell_val = cell_override if cell_override is not None else row.get('Cell', '')
    return No 1st', '0')}{part1.get('Rack No 2nd', '0')}"
        rack_key = f"Rack {rack_num.zfill(2)}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
        elements.append(Spacer(1, 0.2*cm))
        label_count += 1
        
    if elements: doc.build(elements)
    buffer.seek(0)
    return buffer, label_summary

def main():
    st.title("🏷️ Rack Label Generator")
    st.markdown("<p style='font- [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level']] + [str(cell_val)]

# --- Combined PDF Generation ---
def generate_combined_labels(df, progress_bar=None, statusstyle:italic;'>Designed by Agilomatrix</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.sidebar.title("📄 Label Options")
    base_rack_id = st.sidebar.text_input("Enter Storage Line Side Infrastructure", "R")
    
    uploaded_file = st_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    
    df..file_uploader("Choose an Excel or CSV file with a 'Packaging Factor' column", type=['xlsx', 'xls', 'csv'])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, keep_default_na=False) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file, keep_default_na=False)
            st.success(f"✅ Filesort_values(by=['Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell'], inplace=True)
    df_parts_only = df[df['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    
    physical_cell_grouping_cols = ['Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']
    df_grouped = df_parts_only.groupby(physical_cell_grouping_cols)
    
    total_labels, label loaded! Found {len(df)} rows.")
            
            # Find all required columns using their original names
            part_no_col, desc_col, bus_model_col, station_col, container_col, packaging_count, label_summary = len(df_grouped), 0, {}

    for i, (group_key, group) in enumerate(df_grouped):
        if progress_bar: progress_bar.progress(int((i / total_labels) * 100))
        if status_text: status_text.text(f"Processing Label {i+1}/{total_labels}")

        if label_count > 0 and label__factor_col = find_required_columns(df)
            
            if not packaging_factor_col:
                st.error("❌ Critical Error: A column containing 'Packaging Factor' was not found in your file. This column is required.")
            else:
                st.sidebar.markdown("---")
                st.sidebar.subheadercount % 4 == 0: elements.append(PageBreak())

        part1 = group.iloc[0].to_dict()
        
        if group.iloc[0].get('Packaging Factor') == 0.5 and len(group) > 1:
            part2 = group.iloc[1].to_dict()
            part_("Global Rack Configuration")
                levels = st.multiselect("Active Levels (for all racks)", options=['A','B','C','D','E','F','G','H'], default=['A','B','C','D'])
                num_cells_per_level = st.sidebar.number_input("Number of Physical Cells per Level", min_value=1, value=10, step=1)
                num_racks =table1 = Table([['Part No', format_part_no_v1(part1['Part No'])], ['Description', format_description_v1(part1['Description'])]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
            part_table2 = st.sidebar.number_input("Total Number of Racks", min_value=1, value=4, step=1)
                
                # Use the original container column name to get unique values
                unique_containers = get_unique_containers(df, container_col) if container_col else []
                bin_info_map = {}
                 Table([['Part No', format_part_no_v1(part2['Part No'])], ['Description', format_description_v1(part2['Description'])]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
            
            location_values = extract_location_values(part1, cell_override=part1['Cell'])
            location_st.sidebar.markdown("---")
                st.sidebar.subheader("Container (Bin) Dimensions")
                for container in unique_containers:
                    dim = st.sidebar.text_input(f"Dimensions for {container}", key=f"bindim_{container}", placeholder="e.g., 600x400")
                    bin_info_map[container] = {'dim_str': dim}

                if st.button("🚀 Generate PDF Labels", type="primary"):
                    progress_bar = st.progress(0)
                    status_text =data = [['Line Location'] + location_values]
            col_proportions = [1.8, 2.7, 1.3, 1.3, 1.3, 1.3, 1.3]
            location_widths = [4*cm] + [w * (11*cm) / sum(col_proportions) for w in col_proportions]
            location_table = Table( st.empty()
                    try:
                        rack_configs = {}
                        for i in range(num_racks):location_data, colWidths=location_widths, rowHeights=0.8*cm)
            
            elements.
                            rack_name = f"Rack {i+1:02d}"
                            rack_configs[rack_name] = {'levels': levels, 'cells_per_level': num_cells_per_level}

                        df_physically_assigned = automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text)
                        df_final_labels = assign_sequential_location_ids(df_physically_assigned)
                        
                        if df_final_labels is not None andextend([part_table1, Spacer(1, 0.3*cm), part_table2, Spacer(1, 0.3*cm), location_table])

        else:
            part_table = Table([['Part No', format_part_no_v2(part1['Part No'])], ['Description', format_description(part1['Description'])]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
            location_values = extract_location_values(part1)
            location_ not df_final_labels.empty:
                            pdf_buffer, label_summary = generate_combined_labels(df_final_labels, progress_bar, status_text)
                            
                            if pdf_buffer:
                                total_labels = sum(label_summary.values())
                                status_text.text(f"✅ PDF with {total_labels} labels generated successfully!")
                                file_name = f"{os.path.splitext(uploaded_file.data = [['Line Location'] + location_values]
            col_widths = [1.7, 2.9, 1.3, 1.2, 1.3, 1.3, 1.3]
            location_widths = [4*cm] + [w * (11*cm) / sum(col_widths) for w in col_widths]
            location_table = Table(location_data, colWidths=location_widths, rowHeights=0.9*cm)
            
            elements.extend([part_table, Spacer(1, 0.3*cm), location_table])

        rack_numname)[0]}_labels.pdf"
                                st.download_button(label="📥 Download PDF", data=pdf_buffer.getvalue(), file_name=file_name, mime="application/pdf")

                                if total_labels >  = f"{part1.get('Rack No 1st', '0')}{part1.get('Rack No 2nd', '0')}"
        rack_key = f"Rack {rack_num.zfill(2)}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
        elements.append(Spacer(1, 0.2*cm))
        label_count += 1
        
    if elements: doc.build(elements)
    buffer.seek(0)
0:
                                    st.markdown("---")
                                    st.subheader("📊 Generation Summary")
                                    st.markdown(f"A total of **{total_labels}** labels (one per physical cell) have been generated.")
                                    summary_df = pd.DataFrame(list(label_summary.items()), columns=['Rack', 'Number of Labels'])
                                    st.table(summary_df.sort_values(by='Rack').reset_index(    return buffer, label_summary

# --- Main Application UI ---
def main():
    st.title("🏷️ Rack Label Generator")
    st.markdown("<p style='font-style:italic;'>Designed by Agildrop=True))
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
