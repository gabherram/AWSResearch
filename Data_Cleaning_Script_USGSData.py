
import pandas as pd
import os
import xlsxwriter

def plot_csv_to_excel(workbook, worksheet, csv_file, start_row):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract station number from filename
    station_number = os.path.splitext(os.path.basename(csv_file))[0]
    
    # Check if necessary columns are present
    if 'INDEP' not in df.columns or 'DEP' not in df.columns:
        return start_row, station_number  # Return the current row and station number if columns are missing
    
    # Check if the file is empty (has only headers)
    if df.empty:
        return start_row, station_number  # Return the current row and station number if file is empty
    
    # Extract the required columns
    indep = df['INDEP']
    dep = df['DEP']

    # Write the data to the worksheet
    worksheet.write(start_row, 0, 'DEP')
    worksheet.write(start_row, 1, 'INDEP')

    for i, (d, ind) in enumerate(zip(dep, indep), start=start_row + 1):
        worksheet.write(i, 0, d)
        worksheet.write(i, 1, ind)

    # Create a chart object
    chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})

    # Configure the chart
    chart.add_series({
        'name':       'INDEP vs DEP',
        'categories': [worksheet.name, start_row + 1, 0, i, 0],
        'values':     [worksheet.name, start_row + 1, 1, i, 1],
        'marker':     {'type': 'circle', 'size': 5},
        'trendline':  {
            'type': 'polynomial',
            'order': 2,
            'name': 'Polynomial Trendline',
            'display_equation': True,
            'display_r_squared': True
        }
    })
    chart.add_series({
        'name':       'INDEP vs DEP (Power Trendline)',
        'categories': [worksheet.name, start_row + 1, 0, i, 0],
        'values':     [worksheet.name, start_row + 1, 1, i, 1],
        'marker':     {'type': 'circle', 'size': 5},
        'trendline':  {
            'type': 'power',
            'name': 'Power Trendline',
            'display_equation': True,
            'display_r_squared': True
        }
    })
    chart.add_series({
        'name':       'INDEP vs DEP (Exponential Trendline)',
        'categories': [worksheet.name, start_row + 1, 0, i, 0],
        'values':     [worksheet.name, start_row + 1, 1, i, 1],
        'marker':     {'type': 'circle', 'size': 5},
        'trendline':  {
            'type': 'exponential',
            'name': 'Exponential Trendline',
            'display_equation': True,
            'display_r_squared': True
        }
    })

    chart.set_title({'name': f'Station {station_number} - INDEP vs DEP with Trendlines'})
    chart.set_x_axis({'name': 'DEP'})
    chart.set_y_axis({'name': 'INDEP'})

    # Insert the chart into the worksheet
    worksheet.insert_chart(start_row, 3, chart)

    return i + 3, None  # Return the new starting row and None indicating the file is not empty

def process_files(directory, output_excel_xlsx, empty_files_csv):
    workbook = xlsxwriter.Workbook(output_excel_xlsx)
    worksheet = workbook.add_worksheet()
    row = 0
    empty_stations = []

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            csv_path = os.path.join(directory, filename)
            try:
                row, empty_station = plot_csv_to_excel(workbook, worksheet, csv_path, row)
                if empty_station is not None:
                    empty_stations.append(empty_station)
                row += 3  # Add some space between charts
            except ValueError as e:
                print(f"Error processing {filename}: {e}")
    
    workbook.close()

    # Write empty stations to a CSV file
    if empty_stations:
        pd.DataFrame(empty_stations, columns=["Station ID"]).to_csv(empty_files_csv, index=False)

def main():
    directory = '/Users/gabrielhernandez/Desktop/USGS_RC copy'  # Directory path
    output_excel_xlsx = '/Users/gabrielhernandez/Desktop/USGS_RC copy/all_charts.xlsx'  # Intermediate XLSX file
    empty_files_csv = '/Users/gabrielhernandez/Desktop/USGS_RC copy/empty_files.csv'  # CSV file for empty files
    process_files(directory, output_excel_xlsx, empty_files_csv)
    
    print(f"Processing complete. Results saved to '{output_excel_xlsx}' and '{empty_files_csv}'.")

if __name__ == "__main__":
    main()
