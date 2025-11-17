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
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# --- Page Configuration ---
st.set_page_config(
    page_title="Part Label Generator",
    page_icon="🏷️",
    layout="wide"
)

# --- Style Definitions ---
# Styles for Single-Part (V2) Labels
bold_style_v2 = ParagraphStyle(
    name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=12, spaceBefore=0, spaceAfter=15,
)
desc_style_v2 = ParagraphStyle(
    name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16, spaceBefore=2, spaceAfter=2
)

# Styles for Multi-Part (V1) Labels
bold_style_v1 = ParagraphStyle(
    name='Bold_v1', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=20, spaceBefore=2, spaceAfter=2
)

# --- Formatting Functions ---

# --- Functions for Single-Part (V2) Labels ---
def format_part_no_v2(part_no):
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if part_no.upper() == 'EMPTY':
         return Paragraph(f"<b><font size=34>EMPTY</font></b><br/><br/>", bold_style_v2)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b><br/><br/>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b><br/><br/>", bold_style_v2)

def format_description_v2(desc):
    if not desc or not isinstance(desc, str): desc = str(desc)
    return Paragraph(desc, desc_style_v2)

# --- Functions for Multi-Part (V1) Labels ---
def format_part_no_v1(part_no):
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=17>{part1}</font><font size=22>{part2}</font></b>", bold_style_v1)
    return Paragraph(f"<b><font size=17>{part_no}</font></b>", bold_style_v1)

def format_description_v1(desc):
    if not desc or not isinstance(desc, str): desc = str(desc)
    desc_length = len(desc)
    if desc_length <= 30: font_size = 15
    elif desc_length <= 50: font_size = 13
    elif desc_length <= 70: font_size = 11
    elif desc_length <= 90: font_size = 10
    else: font_size = 9
    desc_style_v1 = ParagraphStyle(
        name='Description_v1', fontName='Helvetica', fontSize=font_size, alignment=TA_LEFT,
        leading=font_size + 2, spaceBefore=1, spaceAfter=1
    )
    return Paragraph(desc, desc_style_v1)

# --- Core Logic Functions ---
def find_required_columns(df):
    """Finds all necessary columns, including the new Packing Factor."""
    cols = {col.upper().strip(): col for col in df.columns}
    part_no_key = next((k for k in cols if 'PART' in k and ('NO' in k or 'NUM' in k)), None)
    desc_key = next((k for k in cols if 'DESC' in k), None)
    bus_model_key = next((k for k in cols if 'BUS' in k and 'MODEL' in k), None)
    station_no_key = next((k for k in cols if 'STATION' in k), None)
    container_type_key = next((k for k in cols if 'CONTAINER' in k), None)
    packing_factor_key = next((k for k in cols if 'PACKING' in k and 'FACTOR' in k), None)
    return (cols.get(part_no_key), cols.get(desc_key), cols.get(bus_model_key),
            cols.get(station_no_key), cols.get(container_type_key), cols.get(packing_factor_key))

def get_unique_containers(df, container_col):
    if not container_col or container_col not in df.columns: return []
    return sorted(df[container_col].dropna().astype(str).unique())

def parse_dimensions(dim_str):
    if not isinstance(dim_str, str) or not dim_str: return 0, 0
    nums = [int(n) for n in re.findall(r'\d+', dim_str)]
    return (nums[0], nums[1]) if len(nums) >= 2 else (0, 0)

def automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text=None):
    """Automates location assignment, now considering Packing Factor."""
    part_no_col, desc_col, model_col, station_col, container_col, packing_factor_col = find_required_columns(df)
    if not all([part_no_col, station_col]):
        st.error("❌ 'Part Number' or 'Station No' column not found.")
        return None

    df_processed = df.copy()
    rename_dict = {
        part_no_col: 'Part No', desc_col: 'Description', model_col: 'Bus Model',
        station_col: 'Station No', container_col: 'Container', packing_factor_col: 'Packing Factor'
    }
    df_processed.rename(columns={k: v for k, v in rename_dict.items() if k}, inplace=True)

    if 'Packing Factor' in df_processed.columns:
        df_processed['Packing Factor'] = pd.to_numeric(df_processed['Packing Factor'], errors='coerce').fillna(1.0)
    else:
        st.info("ℹ️ 'Packing Factor' column not found. Defaulting to 1 part per location.")
        df_processed['Packing Factor'] = 1.0

    if container_col:
        df_processed['bin_info'] = df_processed['Container'].map(bin_info_map)
        df_processed['bin_area'] = df_processed['bin_info'].apply(lambda x: x['dims'][0] * x['dims'][1] if x and x.get('dims') else 0)
        df_processed['bins_per_cell'] = df_processed['bin_info'].apply(lambda x: x['capacity'] if x else 1) # Default capacity to 1
    else:
        df_processed['bin_area'] = 0
        df_processed['bins_per_cell'] = 1


    final_df_parts = []
    available_cells = []
    for rack_name, config in sorted(rack_configs.items()):
        rack_num_val = ''.join(filter(str.isdigit, rack_name))
        rack_num_1st = rack_num_val[0] if len(rack_num_val) > 1 else '0'
        rack_num_2nd = rack_num_val[1] if len(rack_num_val) > 1 else rack_num_val[0]
        for level in sorted(config.get('levels', [])):
            for i in range(config.get('cells_per_level', 0)):
                location = {'Level': level, 'Physical_Cell': f"{i + 1:02d}", 'Rack': base_rack_id, 'Rack No 1st': rack_num_1st, 'Rack No 2nd': rack_num_2nd}
                available_cells.append(location)

    current_cell_index = 0
    last_processed_station = "N/A"

    for station_no, station_group in df_processed.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Processing station: {station_no}...")
        last_processed_station = station_no

        # Process parts with Packing Factor = 0.5 first (multi-part per location)
        group_pf_0_5 = station_group[station_group['Packing Factor'] == 0.5]
        if not group_pf_0_5.empty:
            parts_pf_0_5 = group_pf_0_5.to_dict('records')
            for i in range(0, len(parts_pf_0_5), 2):
                if current_cell_index >= len(available_cells): break
                chunk = parts_pf_0_5[i:i + 2]
                current_location = available_cells[current_cell_index]
                for part in chunk:
                    part.update(current_location)
                    final_df_parts.append(part)
                current_cell_index += 1

        # Process all other parts (single-part per location)
        group_pf_1 = station_group[station_group['Packing Factor'] != 0.5]
        parts_grouped_by_container = group_pf_1.groupby('Container' if container_col else 'Part No')
        sorted_groups = sorted(parts_grouped_by_container, key=lambda x: x[1]['bin_area'].iloc[0], reverse=True)

        for _, group_df in sorted_groups:
            parts_to_assign = group_df.to_dict('records')
            bins_per_cell = parts_to_assign[0].get('bins_per_cell', 1)
            if bins_per_cell == 0: continue

            for i in range(0, len(parts_to_assign), bins_per_cell):
                if current_cell_index >= len(available_cells): break
                chunk = parts_to_assign[i:i + bins_per_cell]
                current_location = available_cells[current_cell_index]
                for part in chunk:
                    part.update(current_location)
                    final_df_parts.append(part)
                current_cell_index += 1
            if current_cell_index >= len(available_cells): break
        if current_cell_index >= len(available_cells):
            st.warning("⚠️ All available rack space has been filled.")
            break

    for i in range(current_cell_index, len(available_cells)):
        empty_part = {'Part No': 'EMPTY', 'Description': '', 'Bus Model': '', 'Station No': last_processed_station, 'Container': ''}
        empty_part.update(available_cells[i])
        final_df_parts.append(empty_part)

    return pd.DataFrame(final_df_parts) if final_df_parts else pd.DataFrame()

def assign_sequential_location_ids(df):
    """Gives each PHYSICAL CELL a unique sequential ID that resets per rack/level."""
    if df.empty: return df
    df_sorted = df.sort_values(by=['Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']).copy()
    
    df_parts_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    if df_parts_only.empty:
        df['Cell'] = df['Physical_Cell']
        return df

    df_parts_only['physical_location_key'] = df_parts_only['Rack No 1st'].astype(str) + '-' + \
                                             df_parts_only['Rack No 2nd'].astype(str) + '-' + \
                                             df_parts_only['Level'].astype(str) + '-' + \
                                             df_parts_only['Physical_Cell'].astype(str)
    
    unique_locations = df_parts_only.drop_duplicates(subset=['physical_location_key'])
    
    location_counters = {}
    location_to_cell_map = {}
    
    for _, row in unique_locations.iterrows():
        rack_id = (row['Rack No 1st'], row['Rack No 2nd'])
        level = row['Level']
        counter_key = (rack_id, level)
        
        location_counters[counter_key] = location_counters.get(counter_key, 0) + 1
        location_to_cell_map[row['physical_location_key']] = location_counters[counter_key]
        
    df_parts_only['Cell'] = df_parts_only['physical_location_key'].map(location_to_cell_map)
    
    df_empty_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() == 'EMPTY'].copy()
    df_empty_only['Cell'] = df_empty_only['Physical_Cell']

    return pd.concat([df_parts_only, df_empty_only], ignore_index=True)

def extract_location_values(row):
    """Extracts values for the label, using the final 'Cell' number."""
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]

# --- PDF Generation Functions ---
def generate_labels_from_excel(df, progress_bar=None, status_text=None):
    """Generates PDF with dynamic templates for single or multi-part labels."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    
    df.sort_values(by=['Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True, na_position='last')
    df_parts_only = df[df['Part No'].astype(str).str.upper() != 'EMPTY'].copy()

    if df_parts_only.empty:
        st.warning("No valid parts found to generate labels.")
        return None, {}
        
    df_parts_only['final_location_key'] = df_parts_only['Rack No 1st'].astype(str) + '-' + \
                                          df_parts_only['Rack No 2nd'].astype(str) + '-' + \
                                          df_parts_only['Level'].astype(str) + '-' + \
                                          str(df_parts_only['Cell'].iloc[0] if 'Cell' in df_parts_only.columns else '0')


    grouped_by_location = df_parts_only.groupby('final_location_key')

    total_labels = len(grouped_by_location)
    label_count = 0
    label_summary = {}

    for i, (_, group) in enumerate(grouped_by_location):
        if progress_bar: progress_bar.progress(int((i / total_labels) * 100))
        if status_text: status_text.text(f"Processing Label {i+1}/{total_labels}")

        first_part = group.iloc[0]
        rack_num = f"{first_part.get('Rack No 1st', '0')}{first_part.get('Rack No 2nd', '0')}"
        rack_key = f"Rack {rack_num.zfill(2)}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
            
        if label_count > 0 and label_count % 4 == 0:
            elements.append(PageBreak())

        location_values = extract_location_values(first_part)

        # --- DYNAMICALLY CHOOSE LABEL TYPE ---
        if len(group) >= 2:
            # --- MULTI-PART LABEL ---
            part1, part2 = group.iloc[0], group.iloc[1]
            part_table1 = Table([['Part No', format_part_no_v1(str(part1['Part No']))],['Description', format_description_v1(str(part1['Description']))]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
            part_table2 = Table([['Part No', format_part_no_v1(str(part2['Part No']))],['Description', format_description_v1(str(part2['Description']))]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
            
            style = TableStyle([('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (0, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('LEFTPADDING', (1, 0), (1, -1), 5), ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (0, -1), 16)])
            part_table1.setStyle(style)
            part_table2.setStyle(style)

            location_data = [['Part Location'] + location_values]
            col_props = [1.8, 2.7, 1.3, 1.3, 1.3, 1.3, 1.3]
            location_widths = [4*cm] + [w * (11*cm) / sum(col_props) for w in col_props]
            location_table = Table(location_data, colWidths=location_widths, rowHeights=0.8*cm)
            
            elements.append(part_table1)
            elements.append(Spacer(1, 0.1 * cm))
            elements.append(part_table2)
            elements.append(Spacer(1, 0.1 * cm))

        else:
            # --- SINGLE-PART LABEL ---
            part_table = Table([['Part No', format_part_no_v2(str(first_part['Part No']))], ['Description', format_description_v2(str(first_part['Description']))]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
            part_table.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (0, -1), 'CENTER'), ('ALIGN', (1, 1), (1, -1), 'LEFT'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('LEFTPADDING', (0, 0), (-1, -1), 5), ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (0, -1), 16)]))

            location_data = [['Line Location'] + location_values]
            col_widths = [1.7, 2.9, 1.3, 1.2, 1.3, 1.3, 1.3]
            location_widths = [4 * cm] + [w * (11 * cm) / sum(col_widths) for w in col_widths]
            location_table = Table(location_data, colWidths=location_widths, rowHeights=0.9*cm)
            elements.append(part_table)
            elements.append(Spacer(1, 0.3 * cm))

        # Common location table styling
        location_colors = [colors.HexColor('#E9967A'), colors.HexColor('#ADD8E6'), colors.HexColor('#90EE90'), colors.HexColor('#FFD700'), colors.HexColor('#ADD8E6'), colors.HexColor('#E9967A'), colors.HexColor('#90EE90')]
        location_style_cmds = [('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1,-1), 16)]
        for j, color in enumerate(location_colors): location_style_cmds.append(('BACKGROUND', (j+1, 0), (j+1, 0), color))
        location_table.setStyle(TableStyle(location_style_cmds))

        elements.append(location_table)
        elements.append(Spacer(1, 0.2 * cm))
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
    
    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=['xlsx', 'xls', 'csv'])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.success(f"✅ File loaded! Found {len(df)} rows.")
            
            _, _, _, _, container_col, packing_factor_col = find_required_columns(df)
            
            st.sidebar.markdown("---")
            st.sidebar.subheader("Global Rack Configuration")
            levels = st.sidebar.multiselect("Active Levels (for all racks)", options=['A','B','C','D','E','F','G','H'], default=['A','B','C','D'])
            num_cells_per_level = st.sidebar.number_input("Number of Physical Cells per Level", min_value=1, value=10, step=1)
            num_racks = st.sidebar.number_input("Total Number of Racks", min_value=1, value=4, step=1)
            
            bin_info_map = {}
            if container_col:
                unique_containers = get_unique_containers(df, container_col)
                st.sidebar.markdown("---")
                st.sidebar.subheader("Container (Bin) Rules")
                for container in unique_containers:
                    st.sidebar.markdown(f"**Settings for {container}**")
                    dim = st.sidebar.text_input(f"Dimensions", key=f"bindim_{container}", placeholder="e.g., 600x400 (for sorting)")
                    capacity = st.sidebar.number_input("Parts per Physical Cell (Capacity)", min_value=0, value=1, step=1, key=f"bincap_{container}")
                    bin_info_map[container] = {'dims': parse_dimensions(dim), 'capacity': capacity}

            if st.button("🚀 Generate PDF Labels", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                try:
                    rack_configs = {f"Rack {i+1:02d}": {'levels': levels, 'cells_per_level': num_cells_per_level} for i in range(num_racks)}

                    df_physically_assigned = automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text)
                    df_final_labels = assign_sequential_location_ids(df_physically_assigned)
                    
                    if df_final_labels is not None and not df_final_labels.empty:
                        pdf_buffer, label_summary = generate_labels_from_excel(df_final_labels, progress_bar, status_text)
                        
                        if pdf_buffer:
                            total_labels_generated = sum(label_summary.values())
                            status_text.text(f"✅ PDF with {total_labels_generated} labels generated successfully!")
                            file_name = f"{os.path.splitext(uploaded_file.name)[0]}_labels.pdf"
                            st.download_button(label="📥 Download PDF", data=pdf_buffer.getvalue(), file_name=file_name, mime="application/pdf")

                            if total_labels_generated > 0:
                                st.markdown("---")
                                st.subheader("📊 Generation Summary")
                                st.markdown(f"A total of **{total_labels_generated}** labels have been generated. Here is the breakdown by rack:")
                                summary_df = pd.DataFrame(list(label_summary.items()), columns=['Rack', 'Number of Labels']).sort_values(by='Rack').reset_index(drop=True)
                                st.table(summary_df)
                    else:
                        st.error("❌ No data was processed. Check your input file and rack configurations.")
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
