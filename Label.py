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

# --- Style Definitions (Unchanged) ---
bold_style_v1 = ParagraphStyle(
    name='Bold_v1', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=14, spaceBefore=2, spaceAfter=2
)
bold_style_v2 = ParagraphStyle(
    name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=12, spaceBefore=0, spaceAfter=15,
)
desc_style = ParagraphStyle(
    name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16, spaceBefore=2, spaceAfter=2
)

# --- Formatting Functions (Unchanged) ---
def format_part_no_v1(part_no):
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=17>{part1}</font><font size=22>{part2}</font></b>", bold_style_v1)
    return Paragraph(f"<b><font size=17>{part_no}</font></b>", bold_style_v1)

def format_part_no_v2(part_no):
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if part_no.upper() == 'EMPTY':
         return Paragraph(f"<b><font size=34>EMPTY</font></b><br/><br/>", bold_style_v2)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b><br/><br/>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b><br/><br/>", bold_style_v2)

def format_description_v1(desc):
    if not desc or not isinstance(desc, str): desc = str(desc)
    font_size = 15 if len(desc) <= 30 else 13 if len(desc) <= 50 else 11 if len(desc) <= 70 else 10 if len(desc) <= 90 else 9
    desc_style_v1 = ParagraphStyle(name='Description_v1', fontName='Helvetica', fontSize=font_size, alignment=TA_LEFT, leading=font_size + 2)
    return Paragraph(desc, desc_style_v1)

def format_description(desc):
    if not desc or not isinstance(desc, str): desc = str(desc)
    return Paragraph(desc, desc_style)

# --- Core Logic Functions (Updated) ---
def find_required_columns(df):
    cols = {col.upper().strip(): col for col in df.columns}
    part_no_key = next((k for k in cols if 'PART' in k and ('NO' in k or 'NUM' in k)), None)
    desc_key = next((k for k in cols if 'DESC' in k), None)
    bus_model_key = next((k for k in cols if 'BUS' in k and 'MODEL' in k), None)
    station_no_key = next((k for k in cols if 'STATION' in k), None)
    container_type_key = next((k for k in cols if 'CONTAINER' in k), None)
    return (cols.get(part_no_key), cols.get(desc_key), cols.get(bus_model_key),
            cols.get(station_no_key), cols.get(container_type_key))

def get_unique_containers(df, container_col):
    if not container_col or container_col not in df.columns: return []
    return sorted(df[container_col].dropna().astype(str).unique())

def parse_dimensions(dim_str):
    if not isinstance(dim_str, str) or not dim_str: return 0, 0
    nums = [int(n) for n in re.findall(r'\d+', dim_str)]
    return (nums[0], nums[1]) if len(nums) >= 2 else (0, 0)

def can_place(cell, w, d):
    """Finds a valid (x, y) position for a new box of size (w, d) in a cell."""
    for node in cell['nodes']:
        x, y = node['x'], node['y']
        if x + w > cell['width'] or y + d > cell['depth']:
            continue
        
        is_valid_spot = True
        for placed_box in cell['placed_boxes']:
            # Check for overlap
            if not (x + w <= placed_box['x'] or x >= placed_box['x'] + placed_box['w'] or \
                    y + d <= placed_box['y'] or y >= placed_box['y'] + placed_box['d']):
                is_valid_spot = False
                break
        if is_valid_spot:
            return {'x': x, 'y': y}
    return None

def add_part_to_cell(cell, part, x, y, w, d):
    """Adds a part to a cell and updates the placement nodes."""
    cell['parts'].append(part)
    cell['placed_boxes'].append({'x': x, 'y': y, 'w': w, 'd': d})
    
    # Remove the node that was just used
    cell['nodes'] = [n for n in cell['nodes'] if not (n['x'] == x and n['y'] == y)]
    
    # Add two new potential nodes
    new_nodes = [{'x': x + w, 'y': y}, {'x': x, 'y': y + d}]
    for new_node in new_nodes:
        # Only add if it's within bounds and not already a corner of another box
        is_redundant = False
        if new_node['x'] >= cell['width'] or new_node['y'] >= cell['depth']:
            continue
        for existing_node in cell['nodes']:
            if existing_node['x'] == new_node['x'] and existing_node['y'] == new_node['y']:
                is_redundant = True
                break
        if not is_redundant:
            cell['nodes'].append(new_node)
    
    # Sort nodes to prioritize top-left placement
    cell['nodes'].sort(key=lambda item: (item['y'], item['x']))

def automate_location_assignment(df, base_rack_id, rack_configs, bin_dimensions_map, status_text=None):
    """Performs 2D bin packing to assign parts to locations."""
    part_no_col, desc_col, model_col, station_col, container_col = find_required_columns(df)
    if not all([part_no_col, container_col, station_col]):
        st.error("❌ 'Part Number', 'Container Type', or 'Station No' column not found.")
        return None

    df_processed = df.copy()
    rename_dict = {'Part No': 'Part No', 'Description': 'Description', 'Bus Model': 'Bus Model', 'Station No': 'Station No', 'Container': 'Container'}
    df_processed.rename(columns={find_required_columns(df)[i]: v for i, v in enumerate(rename_dict.values())}, inplace=True)
    
    df_processed['bin_width'], df_processed['bin_depth'] = zip(*df_processed['Container'].map(lambda c: parse_dimensions(bin_dimensions_map.get(c, ''))))
    df_processed['bin_area'] = df_processed['bin_width'] * df_processed['bin_depth']
    df_processed.sort_values(by=['Station No', 'bin_area'], ascending=[True, False], inplace=True)

    final_df_parts = []
    
    for station_no, station_group in df_processed.groupby('Station No', sort=False):
        if status_text: status_text.text(f"Processing station: {station_no}...")
        
        available_cells = []
        for rack_name, config in sorted(rack_configs.items()):
            rack_num_val = ''.join(filter(str.isdigit, rack_name))
            rack_num_1st, rack_num_2nd = (rack_num_val[0], rack_num_val[1]) if len(rack_num_val) > 1 else ('0', rack_num_val[0])
            
            cell_width, cell_depth = parse_dimensions(config.get('cell_dimensions', ''))
            
            for level in sorted(config.get('levels', [])):
                for i in range(config.get('cells_per_level', 0)):
                    cell = {
                        'location': {'Level': level, 'Cell': f"{i + 1:02d}", 'Rack': base_rack_id, 'Rack No 1st': rack_num_1st, 'Rack No 2nd': rack_num_2nd},
                        'width': cell_width, 'depth': cell_depth,
                        'parts': [], 'placed_boxes': [], 'nodes': [{'x': 0, 'y': 0}]
                    }
                    available_cells.append(cell)

        parts_to_assign = station_group.to_dict('records')
        unassigned_parts = []
        
        for part in parts_to_assign:
            bin_w, bin_d = part['bin_width'], part['bin_depth']
            if bin_w == 0 or bin_d == 0:
                unassigned_parts.append(part)
                continue

            part_placed = False
            for cell in available_cells:
                # Try original orientation
                pos = can_place(cell, bin_w, bin_d)
                if pos:
                    add_part_to_cell(cell, part, pos['x'], pos['y'], bin_w, bin_d)
                    part_placed = True
                    break
                
                # Try rotated orientation
                pos = can_place(cell, bin_d, bin_w)
                if pos:
                    add_part_to_cell(cell, part, pos['x'], pos['y'], bin_d, bin_w)
                    part_placed = True
                    break
            
            if not part_placed:
                unassigned_parts.append(part)

        if unassigned_parts:
            st.warning(f"⚠️ For station {station_no}, could not assign locations for {len(unassigned_parts)} parts due to insufficient capacity.")

        # Deconstruct the packed cells into the final flat DataFrame
        for cell in available_cells:
            if cell['parts']:
                for part_record in cell['parts']:
                    # Augment original part data with the final location
                    part_record.update(cell['location'])
                    final_df_parts.append(part_record)
            else:
                empty_part = {'Part No': 'EMPTY', 'Description': '', 'Bus Model': '', 'Station No': station_no, 'Container': ''}
                empty_part.update(cell['location'])
                final_df_parts.append(empty_part)

    return pd.DataFrame(final_df_parts) if final_df_parts else pd.DataFrame()


def create_location_key(row):
    return '_'.join([str(row.get(c, '')) for c in ['Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']])

def extract_location_values(row):
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]

# --- PDF Generation Functions (Unchanged) ---
def generate_labels_from_excel_v1(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    
    df['location_key'] = df.apply(create_location_key, axis=1)
    df.sort_values(by=['Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    # Group by the unique location key to generate one label per packed cell
    df_grouped = df.groupby('location_key')
    total_locations = len(df_grouped)
    label_count = 0
    label_summary = {}

    for i, (location_key, group) in enumerate(df_grouped):
        if progress_bar: progress_bar.progress(int((i / total_locations) * 100))
        if status_text: status_text.text(f"Processing V1 Label {i+1}/{total_locations}")
        
        part1 = group.iloc[0]
        if str(part1['Part No']).upper() == 'EMPTY':
            continue

        rack_num = f"{part1.get('Rack No 1st', '0')}{part1.get('Rack No 2nd', '0')}"
        rack_key = f"Rack {rack_num.zfill(2)}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1

        if label_count > 0 and label_count % 4 == 0: elements.append(PageBreak())
        
        # In this format, we show the first two parts placed in the location
        part2 = group.iloc[1] if len(group) > 1 else part1
        
        part_table1 = Table([['Part No', format_part_no_v1(str(part1['Part No']))], ['Description', format_description_v1(str(part1['Description']))]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        part_table2 = Table([['Part No', format_part_no_v1(str(part2['Part No']))], ['Description', format_description_v1(str(part2['Description']))]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        
        location_values = extract_location_values(part1)
        location_data = [['Line Location'] + location_values]
        col_proportions = [1.8, 2.7, 1.3, 1.3, 1.3, 1.3, 1.3]
        location_widths = [4 * cm] + [w * (11 * cm) / sum(col_proportions) for w in col_proportions]
        location_table = Table(location_data, colWidths=location_widths, rowHeights=0.8*cm)
        
        part_style = TableStyle([('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (0, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (0, -1), 16)])
        part_table1.setStyle(part_style)
        part_table2.setStyle(part_style)
        
        location_colors = [colors.HexColor('#E9967A'), colors.HexColor('#ADD8E6'), colors.HexColor('#90EE90'), colors.HexColor('#FFD700'), colors.HexColor('#ADD8E6'), colors.HexColor('#E9967A'), colors.HexColor('#90EE90')]
        location_style = [('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (0, 0), 16), ('FONTSIZE', (1, 0), (-1, -1), 14)]
        for j, color in enumerate(location_colors): location_style.append(('BACKGROUND', (j+1, 0), (j+1, 0), color))
        location_table.setStyle(TableStyle(location_style))
        
        elements.append(part_table1)
        elements.append(Spacer(1, 0.3 * cm))
        elements.append(part_table2)
        elements.append(Spacer(1, 0.3 * cm))
        elements.append(location_table)
        elements.append(Spacer(1, 0.2 * cm))
        label_count += 1
        
    if elements: doc.build(elements)
    buffer.seek(0)
    return buffer, label_summary

def generate_labels_from_excel_v2(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    
    df['location_key'] = df.apply(create_location_key, axis=1)
    df.sort_values(by=['Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    df_grouped = df.groupby('location_key')
    total_locations = len(df_grouped)
    label_count = 0
    label_summary = {}

    for i, (location_key, group) in enumerate(df_grouped):
        if progress_bar: progress_bar.progress(int((i / total_locations) * 100))
        if status_text: status_text.text(f"Processing V2 Label {i+1}/{total_locations}")

        part1 = group.iloc[0]
        if str(part1['Part No']).upper() == 'EMPTY':
            continue
        
        rack_num = f"{part1.get('Rack No 1st', '0')}{part1.get('Rack No 2nd', '0')}"
        rack_key = f"Rack {rack_num.zfill(2)}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
            
        if label_count > 0 and label_count % 4 == 0: elements.append(PageBreak())

        part_table = Table([['Part No', format_part_no_v2(str(part1['Part No']))], ['Description', format_description(str(part1['Description']))]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        
        location_values = extract_location_values(part1)
        location_data = [['Line Location'] + location_values]
        col_widths = [1.7, 2.9, 1.3, 1.2, 1.3, 1.3, 1.3]
        location_widths = [4 * cm] + [w * (11 * cm) / sum(col_widths) for w in col_widths]
        location_table = Table(location_data, colWidths=location_widths, rowHeights=0.9*cm)
        
        part_table.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (0, -1), 'CENTER'), ('ALIGN', (1, 1), (1, -1), 'LEFT'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('LEFTPADDING', (0, 0), (-1, -1), 5), ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (0, -1), 16)]))
        
        location_colors = [colors.HexColor('#E9967A'), colors.HexColor('#ADD8E6'), colors.HexColor('#90EE90'), colors.HexColor('#FFD700'), colors.HexColor('#ADD8E6'), colors.HexColor('#E9967A'), colors.HexColor('#90EE90')]
        location_style = [('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (0, 0), 16), ('FONTSIZE', (1, 0), (-1, -1), 16)]
        for j, color in enumerate(location_colors): location_style.append(('BACKGROUND', (j+1, 0), (j+1, 0), color))
        location_table.setStyle(TableStyle(location_style))
        
        elements.append(part_table)
        elements.append(Spacer(1, 0.3 * cm))
        elements.append(location_table)
        elements.append(Spacer(1, 0.2 * cm))
        label_count += 1
        
    if elements: doc.build(elements)
    buffer.seek(0)
    return buffer, label_summary


# --- Main Application UI (Unchanged) ---
def main():
    st.title("🏷️ Rack Label Generator")
    st.markdown("<p style='font-style:italic;'>Designed by Agilomatrix</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.sidebar.title("📄 Label Options")
    label_type = st.sidebar.selectbox("Choose Label Format:", ["Single Part", "Multiple Parts"])
    base_rack_id = st.sidebar.text_input("Enter Storage Line Side Infrastructure", "R")
    
    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=['xlsx', 'xls', 'csv'])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.success(f"✅ File loaded! Found {len(df)} rows.")
            
            _, _, _, _, container_col = find_required_columns(df)
            
            if container_col:
                st.sidebar.markdown("---")
                st.sidebar.subheader("Global Rack Configuration")
                
                cell_dim = st.sidebar.text_input("Cell Dimensions (for all racks)", placeholder="e.g., 800x400")
                levels = st.sidebar.multiselect("Active Levels (for all racks)", options=['A','B','C','D','E','F','G','H'], default=['A','B','C','D'])
                num_cells_per_level = st.sidebar.number_input("Number of Cells per Level", min_value=1, value=10, step=1)
                num_racks = st.sidebar.number_input("Total Number of Racks", min_value=1, value=4, step=1)
                
                unique_containers = get_unique_containers(df, container_col)
                bin_dimensions_map = {}
                st.sidebar.markdown("---")
                st.sidebar.subheader("Container (Bin) Dimensions")
                for container in unique_containers:
                    dim = st.sidebar.text_input(f"Dimensions for {container}", key=f"bindim_{container}", placeholder="e.g., 600x400")
                    bin_dimensions_map[container] = dim

                if st.button("🚀 Generate PDF Labels", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    try:
                        rack_configs = {}
                        for i in range(num_racks):
                            rack_name = f"Rack {i+1:02d}"
                            rack_configs[rack_name] = {
                                'cell_dimensions': cell_dim, 'levels': levels, 'cells_per_level': num_cells_per_level
                            }

                        df_processed = automate_location_assignment(df, base_rack_id, rack_configs, bin_dimensions_map, status_text)
                        
                        if df_processed is not None and not df_processed.empty:
                            gen_func = generate_labels_from_excel_v2 if label_type == "Single Part" else generate_labels_from_excel_v1
                            pdf_buffer, label_summary = gen_func(df_processed, progress_bar, status_text)
                            
                            if pdf_buffer:
                                total_labels = sum(label_summary.values())
                                status_text.text(f"✅ PDF with {total_labels} labels generated successfully!")
                                file_name = f"{os.path.splitext(uploaded_file.name)[0]}_{label_type.lower().replace(' ','_')}_labels.pdf"
                                st.download_button(label="📥 Download PDF", data=pdf_buffer.getvalue(), file_name=file_name, mime="application/pdf")

                                if total_labels > 0:
                                    st.markdown("---")
                                    st.subheader("📊 Generation Summary")
                                    st.markdown(f"A total of **{total_labels}** labels have been generated. Here is the breakdown by location:")
                                    summary_df = pd.DataFrame(list(label_summary.items()), columns=['Rack', 'Number of Labels'])
                                    summary_df = summary_df.sort_values(by='Rack').reset_index(drop=True)
                                    st.table(summary_df)
                        else:
                            st.error("❌ No data was processed. Check your input file and rack configurations.")
                    except Exception as e:
                        st.error(f"❌ An unexpected error occurred: {e}")
                        st.exception(e)
                    finally:
                        progress_bar.empty()
                        status_text.empty()
            else:
                st.error("Could not find a 'Container Type' column in the file.")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    else:
        st.info("👆 Upload a file to begin.")

if __name__ == "__main__":
    main()
