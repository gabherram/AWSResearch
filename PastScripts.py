#Scripts that didn't work

# import os
# import pandas as pd
# import numpy as np
# from scipy.optimize import curve_fit
# import re


# # Take out conversion
# # Conversion factors
# FEET_TO_METERS = 0.3048
# CUBIC_FEET_PER_SECOND_TO_CUBIC_METERS_PER_SECOND = 0.0283168

# # Function to fit different models and calculate R-squared
# def calculate_trendlines(x, y):
#     results = {}

#     # Power Trendline (y = a * x^b)
#     def power_law(x, a, b):
#         return a * np.power(x, b)

#     try:
#         popt, _ = curve_fit(power_law, x, y, maxfev=10000)
#         y_pred = power_law(x, *popt)
#         residuals = y - y_pred
#         ss_res = np.sum(residuals**2)
#         ss_tot = np.sum((y - np.mean(y))**2)
#         r_squared = 1 - (ss_res / ss_tot)
#         results['Power'] = (r_squared, popt)
#     except Exception as e:
#         print(f"Power model fitting failed: {e}")
#         results['Power'] = (None, None)

#     # Exponential Trendline (y = a * exp(b * x))
#     def exponential(x, a, b):
#         return a * np.exp(b * x)

#     # Remove zero values in y (DEP)
#     non_zero_mask = y > 0
#     x_non_zero = x[non_zero_mask]
#     y_non_zero = y[non_zero_mask]

#     # Log transformation of y values
#     log_y = np.log(y_non_zero)

#     try:
#         popt, _ = curve_fit(exponential, log_y, x_non_zero, maxfev=10000)
#         y_pred = exponential(log_y, *popt)
#         residuals = x_non_zero - y_pred
#         ss_res = np.sum(residuals**2)
#         ss_tot = np.sum((x_non_zero - np.mean(x_non_zero))**2)
#         r_squared = 1 - (ss_res / ss_tot)
#         results['Exponential'] = (r_squared, popt)
#     except Exception as e:
#         print(f"Exponential model fitting failed: {e}")
#         results['Exponential'] = (None, None)

#     # Polynomial Trendline (y = a * x^2 + b * x + c)
#     try:
#         poly_coeffs = np.polyfit(x, y, 2)
#         poly_eq = np.poly1d(poly_coeffs)
#         y_pred = poly_eq(x)
#         residuals = y - y_pred
#         ss_res = np.sum(residuals**2)
#         ss_tot = np.sum((y - np.mean(y))**2)
#         r_squared = 1 - (ss_res / ss_tot)
#         results['Polynomial'] = (r_squared, poly_coeffs)
#     except Exception as e:
#         print(f"Polynomial model fitting failed: {e}")
#         results['Polynomial'] = (None, None)

#     return results

# # Main processing function
# def process_files(directory):
#     results = []
#     no_data_files = []

#     for filename in os.listdir(directory):
#         if filename.endswith(".csv") and filename.startswith("Rating_Data_"):
#             station_id = re.findall(r'\d+', filename)[0]
#             file_path = os.path.join(directory, filename)
#             print(f"Processing file: {file_path}")
#             try:
#                 # Read the file without assuming headers
#                 df = pd.read_csv(file_path, header=None)
#                 if df.shape[0] <= 1:
#                     no_data_files.append((station_id, "No data available"))
#                     continue
                
#                 # Extract headers and data
#                 headers = df.iloc[0]
#                 data = df[1:]
#                 data.columns = headers
                
#                 # Debug: Print headers and first few rows of data
#                 print(f"Headers: {headers.tolist()}")
#                 print(f"Data sample:\n{data.head()}")
                
#                 # Check if required columns are present
#                 if 'INDEP' not in data.columns or 'DEP' not in data.columns:
#                     no_data_files.append((station_id, "Required columns not found"))
#                     continue
                
#                 indep = data['INDEP'].dropna().astype(float).values * FEET_TO_METERS  # Convert to meters
#                 dep = data['DEP'].dropna().astype(float).values * CUBIC_FEET_PER_SECOND_TO_CUBIC_METERS_PER_SECOND  # Convert to m³/s

#                 if len(indep) == 0 or len(dep) == 0:
#                     no_data_files.append((station_id, "No data available"))
#                     continue

#                 trendline_results = calculate_trendlines(dep, indep)
#                 best_fit = max(trendline_results, key=lambda k: trendline_results[k][0] if trendline_results[k][0] is not None else -1)
#                 best_r_squared = trendline_results[best_fit][0]
#                 best_coeffs = trendline_results[best_fit][1]

#                 if best_coeffs is not None and len(best_coeffs) >= 2:
#                     variable_a, variable_b = best_coeffs[0], best_coeffs[1]
#                 else:
#                     variable_a, variable_b = None, None

#                 results.append([station_id, trendline_results['Power'][0], trendline_results['Exponential'][0], trendline_results['Polynomial'][0], best_fit, best_r_squared, best_coeffs, variable_a, variable_b])
#                 print(f"Processed {station_id} with best fit: {best_fit} and R-squared: {best_r_squared}")

#             except Exception as e:
#                 no_data_files.append((station_id, f"Error: {str(e)}"))
#                 print(f"Failed to process {station_id}: {e}")
#                 continue

#     results_df = pd.DataFrame(results, columns=['Station ID', 'Power R-squared', 'Exponential R-squared', 'Polynomial R-squared', 'Best Fit', 'Best R-squared', 'Coefficients', 'Variable a', 'Variable b'])
#     no_data_df = pd.DataFrame(no_data_files, columns=['Station ID', 'Message'])

#     return results_df, no_data_df

# # Run the processing function and save results
# def main():
#     directory = '/Users/gabrielhernandez/Desktop/Data-AWS RESEARCH/USGS_RC copy'  # Replace with your directory path
#     results_df, no_data_df = process_files(directory)
    
#     # Save results in the same directory
#     results_path = os.path.join(directory, 'trendline_results.csv')
#     no_data_path = os.path.join(directory, 'no_data_files.csv')
    
#     results_df.to_csv(results_path, index=False)
#     no_data_df.to_csv(no_data_path, index=False)
    
#     print(f"Processing complete. Results saved to '{results_path}' and '{no_data_path}'.")

# if __name__ == "__main__":
#     main()

# import os
# import pandas as pd
# import numpy as np
# from scipy.optimize import curve_fit
# import re

# # Function to fit different models and calculate R-squared
# def calculate_trendlines(x, y):
#     results = {}

#     # Power Trendline (y = a * x^b)
#     def power_law(x, a, b):
#         return a * np.power(x, b)

#     try:
#         popt, _ = curve_fit(power_law, x, y, maxfev=10000)
#         y_pred = power_law(x, *popt)
#         residuals = y - y_pred
#         ss_res = np.sum(residuals**2)
#         ss_tot = np.sum((y - np.mean(y))**2)
#         r_squared = 1 - (ss_res / ss_tot)
#         results['Power'] = (r_squared, popt)
#     except Exception as e:
#         print(f"Power model fitting failed: {e}")
#         results['Power'] = (None, None)

#     # Exponential Trendline (y = a * exp(b * x))
#     def exponential(x, a, b):
#         return a * np.exp(b * x)

#     # Remove zero values in y (DEP)
#     non_zero_mask = y > 0
#     x_non_zero = x[non_zero_mask]
#     y_non_zero = y[non_zero_mask]

#     # Log transformation of y values
#     log_y = np.log(y_non_zero)

#     try:
#         popt, _ = curve_fit(exponential, log_y, x_non_zero, maxfev=10000)
#         y_pred = exponential(log_y, *popt)
#         residuals = x_non_zero - y_pred
#         ss_res = np.sum(residuals**2)
#         ss_tot = np.sum((x_non_zero - np.mean(x_non_zero))**2)
#         r_squared = 1 - (ss_res / ss_tot)
#         results['Exponential'] = (r_squared, popt)
#     except Exception as e:
#         print(f"Exponential model fitting failed: {e}")
#         results['Exponential'] = (None, None)

#     # Polynomial Trendline (y = a * x^2 + b * x + c)
#     try:
#         poly_coeffs = np.polyfit(x, y, 2)
#         poly_eq = np.poly1d(poly_coeffs)
#         y_pred = poly_eq(x)
#         residuals = y - y_pred
#         ss_res = np.sum(residuals**2)
#         ss_tot = np.sum((y - np.mean(y))**2)
#         r_squared = 1 - (ss_res / ss_tot)
#         results['Polynomial'] = (r_squared, poly_coeffs)
#     except Exception as e:
#         print(f"Polynomial model fitting failed: {e}")
#         results['Polynomial'] = (None, None)

#     return results

# # Main processing function
# def process_files(directory):
#     results = []
#     no_data_files = []

#     for filename in os.listdir(directory):
#         if filename.endswith(".csv") and filename.startswith("Rating_Data_"):
#             station_id = re.findall(r'\d+', filename)[0]
#             file_path = os.path.join(directory, filename)
#             print(f"Processing file: {file_path}")
#             try:
#                 # Read the file without assuming headers
#                 df = pd.read_csv(file_path, header=None)
#                 if df.shape[0] <= 1:
#                     no_data_files.append((station_id, "No data available"))
#                     continue
                
#                 # Extract headers and data
#                 headers = df.iloc[0]
#                 data = df[1:]
#                 data.columns = headers
                
#                 # Debug: Print headers and first few rows of data
#                 print(f"Headers: {headers.tolist()}")
#                 print(f"Data sample:\n{data.head()}")
                
#                 # Check if required columns are present
#                 if 'INDEP' not in data.columns or 'DEP' not in data.columns:
#                     no_data_files.append((station_id, "Required columns not found"))
#                     continue
                
#                 indep = data['INDEP'].dropna().astype(float).values  # No conversion
#                 dep = data['DEP'].dropna().astype(float).values  # No conversion

#                 if len(indep) == 0 or len(dep) == 0:
#                     no_data_files.append((station_id, "No data available"))
#                     continue

#                 trendline_results = calculate_trendlines(dep, indep)
#                 best_fit = max(trendline_results, key=lambda k: trendline_results[k][0] if trendline_results[k][0] is not None else -1)
#                 best_r_squared = trendline_results[best_fit][0]
#                 best_coeffs = trendline_results[best_fit][1]

#                 if best_coeffs is not None and len(best_coeffs) >= 2:
#                     variable_a, variable_b = best_coeffs[0], best_coeffs[1]
#                 else:
#                     variable_a, variable_b = None, None

#                 results.append([station_id, trendline_results['Power'][0], trendline_results['Exponential'][0], trendline_results['Polynomial'][0], best_fit, best_r_squared, best_coeffs, variable_a, variable_b])
#                 print(f"Processed {station_id} with best fit: {best_fit} and R-squared: {best_r_squared}")

#             except Exception as e:
#                 no_data_files.append((station_id, f"Error: {str(e)}"))
#                 print(f"Failed to process {station_id}: {e}")
#                 continue

#     results_df = pd.DataFrame(results, columns=['Station ID', 'Power R-squared', 'Exponential R-squared', 'Polynomial R-squared', 'Best Fit', 'Best R-squared', 'Coefficients', 'Variable a', 'Variable b'])
#     no_data_df = pd.DataFrame(no_data_files, columns=['Station ID', 'Message'])

#     return results_df, no_data_df

# # Run the processing function and save results
# def main():
#     directory = '/Users/gabrielhernandez/Desktop/Data-AWS RESEARCH/USGS_RC copy'  # Replace with your directory path
#     results_df, no_data_df = process_files(directory)
    
#     # Save results in the same directory
#     results_path = os.path.join(directory, 'trendline_results.csv')
#     no_data_path = os.path.join(directory, 'no_data_files.csv')
    
#     results_df.to_csv(results_path, index=False)
#     no_data_df.to_csv(no_data_path, index=False)
    
#     print(f"Processing complete. Results saved to '{results_path}' and '{no_data_path}'.")

# if __name__ == "__main__":
#     main()

# import os
# import pandas as pd
# import numpy as np
# from scipy.optimize import curve_fit
# import re

# # Function to fit different models and calculate R-squared
# def calculate_trendlines(x, y):
#     results = {}

#     # Power Trendline (y = a * x^b)
#     def power_law(x, a, b):
#         return a * np.power(x, b)

#     try:
#         popt, _ = curve_fit(power_law, x, y, maxfev=10000)
#         y_pred = power_law(x, *popt)
#         residuals = y - y_pred
#         ss_res = np.sum(residuals**2)
#         ss_tot = np.sum((y - np.mean(y))**2)
#         r_squared = 1 - (ss_res / ss_tot)
#         results['Power'] = r_squared
#     except Exception as e:
#         print(f"Power model fitting failed: {e}")
#         results['Power'] = None

#     # Exponential Trendline (y = a * exp(b * x))
#     def exponential(x, a, b):
#         return a * np.exp(b * x)

#     # Remove zero values in y (DEP)
#     non_zero_mask = y > 0
#     x_non_zero = x[non_zero_mask]
#     y_non_zero = y[non_zero_mask]

#     # Log transformation of y values
#     log_y = np.log(y_non_zero)

#     try:
#         popt, _ = curve_fit(exponential, log_y, x_non_zero, maxfev=10000)
#         y_pred = exponential(log_y, *popt)
#         residuals = x_non_zero - y_pred
#         ss_res = np.sum(residuals**2)
#         ss_tot = np.sum((x_non_zero - np.mean(x_non_zero))**2)
#         r_squared = 1 - (ss_res / ss_tot)
#         results['Exponential'] = r_squared
#     except Exception as e:
#         print(f"Exponential model fitting failed: {e}")
#         results['Exponential'] = None

#     # Polynomial Trendline (y = a * x^2 + b * x + c)
#     try:
#         poly_coeffs = np.polyfit(x, y, 2)
#         poly_eq = np.poly1d(poly_coeffs)
#         y_pred = poly_eq(x)
#         residuals = y - y_pred
#         ss_res = np.sum(residuals**2)
#         ss_tot = np.sum((y - np.mean(y))**2)
#         r_squared = 1 - (ss_res / ss_tot)
#         results['Polynomial'] = r_squared
#     except Exception as e:
#         print(f"Polynomial model fitting failed: {e}")
#         results['Polynomial'] = None

#     return results

# # Main processing function
# def process_files(directory):
#     results = []
#     no_data_files = []

#     for filename in os.listdir(directory):
#         if filename.endswith(".csv") and filename.startswith("Rating_Data_"):
#             station_id = re.findall(r'\d+', filename)[0]
#             file_path = os.path.join(directory, filename)
#             print(f"Processing file: {file_path}")
#             try:
#                 # Read the file without assuming headers
#                 df = pd.read_csv(file_path, header=None)
#                 if df.shape[0] <= 1:
#                     no_data_files.append((station_id, "No data available"))
#                     continue
                
#                 # Extract headers and data
#                 headers = df.iloc[0]
#                 data = df[1:]
#                 data.columns = headers
                
#                 # Debug: Print headers and first few rows of data
#                 print(f"Headers: {headers.tolist()}")
#                 print(f"Data sample:\n{data.head()}")
                
#                 # Check if required columns are present
#                 if 'INDEP' not in data.columns or 'DEP' not in data.columns:
#                     no_data_files.append((station_id, "Required columns not found"))
#                     continue
                
#                 indep = data['INDEP'].dropna().astype(float).values  # No conversion
#                 dep = data['DEP'].dropna().astype(float).values  # No conversion

#                 if len(indep) == 0 or len(dep) == 0:
#                     no_data_files.append((station_id, "No data available"))
#                     continue

#                 trendline_results = calculate_trendlines(dep, indep)
#                 best_fit = max(trendline_results, key=lambda k: trendline_results[k] if trendline_results[k] is not None else -1)
#                 best_r_squared = trendline_results[best_fit]

#                 results.append([station_id, trendline_results['Power'], trendline_results['Exponential'], trendline_results['Polynomial'], best_fit, best_r_squared])
#                 print(f"Processed {station_id} with best fit: {best_fit} and R-squared: {best_r_squared}")

#             except Exception as e:
#                 no_data_files.append((station_id, f"Error: {str(e)}"))
#                 print(f"Failed to process {station_id}: {e}")
#                 continue

#     results_df = pd.DataFrame(results, columns=['Station ID', 'Power R-squared', 'Exponential R-squared', 'Polynomial R-squared', 'Best Fit', 'Best R-squared'])
#     no_data_df = pd.DataFrame(no_data_files, columns=['Station ID', 'Message'])

#     return results_df, no_data_df

# # Run the processing function and save results
# def main():
#     directory = '/Users/gabrielhernandez/Desktop/Data-AWS RESEARCH/USGS_RC copy'  # Replace with your directory path
#     results_df, no_data_df = process_files(directory)
    
#     # Save results in the same directory
#     results_path = os.path.join(directory, 'trendline_results.csv')
#     no_data_path = os.path.join(directory, 'no_data_files.csv')
    
#     results_df.to_csv(results_path, index=False)
#     no_data_df.to_csv(no_data_path, index=False)
    
#     print(f"Processing complete. Results saved to '{results_path}' and '{no_data_path}'.")

# if __name__ == "__main__":
#     main()

# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import xlsxwriter

# def plot_csv_to_excel(csv_file, excel_file):
#     # Read the CSV file
#     df = pd.read_csv(csv_file)
    
#     # Check if necessary columns are present
#     if 'INDEP' not in df.columns or 'DEP' not in df.columns:
#         raise ValueError(f"CSV file {csv_file} must contain 'INDEP' and 'DEP' columns.")
    
#     # Extract the required columns
#     indep = df['INDEP']
#     dep = df['DEP']
    
#     # Create a new Excel workbook and add a worksheet
#     workbook = xlsxwriter.Workbook(excel_file)
#     worksheet = workbook.add_worksheet()
    
#     # Write the data to the worksheet
#     worksheet.write('A1', 'DEP')
#     worksheet.write('B1', 'INDEP')
    
#     for i, (d, ind) in enumerate(zip(dep, indep), start=1):
#         worksheet.write(i, 0, d)
#         worksheet.write(i, 1, ind)
    
#     # Create a chart object
#     chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})
    
#     # Configure the chart
#     chart.add_series({
#         'name':       'INDEP vs DEP',
#         'categories': [worksheet.name, 1, 0, len(dep), 0],
#         'values':     [worksheet.name, 1, 1, len(indep), 1],
#         'marker':     {'type': 'circle', 'size': 5},
#     })
    
#     chart.set_title({'name': 'INDEP vs DEP'})
#     chart.set_x_axis({'name': 'DEP'})
#     chart.set_y_axis({'name': 'INDEP'})
    
#     # Insert the chart into the worksheet
#     worksheet.insert_chart('D2', chart)
    
#     # Close the workbook
#     workbook.close()

# def process_files(directory):
#     results = []
#     no_data_files = []

#     for filename in os.listdir(directory):
#         if filename.endswith(".csv"):
#             csv_path = os.path.join(directory, filename)
#             excel_path = os.path.join(directory, filename.replace('.csv', '.xlsx'))

#             try:
#                 plot_csv_to_excel(csv_path, excel_path)
#                 results.append((filename, "Success"))
#             except ValueError as e:
#                 print(f"Error processing {filename}: {e}")
#                 no_data_files.append(filename)
    
#     return pd.DataFrame(results, columns=['Filename', 'Status']), pd.DataFrame(no_data_files, columns=['Filename'])

# def main():
#     directory = '/Users/gabrielhernandez/Desktop/Test Code'  # Directory path
#     results_df, no_data_df = process_files(directory)
    
#     # Save results in the same directory
#     results_path = os.path.join(directory, 'trendline_results.csv')
#     no_data_path = os.path.join(directory, 'no_data_files.csv')
    
#     results_df.to_csv(results_path, index=False)
#     no_data_df.to_csv(no_data_path, index=False)
    
#     print(f"Processing complete. Results saved to '{results_path}' and '{no_data_path}'.")

# if __name__ == "__main__":
#     main()

# import pandas as pd
# import os
# import xlsxwriter

# def plot_csv_to_excel(csv_file, excel_file):
#     # Read the CSV file
#     df = pd.read_csv(csv_file)
    
#     # Check if necessary columns are present
#     if 'INDEP' not in df.columns or 'DEP' not in df.columns:
#         raise ValueError(f"CSV file {csv_file} must contain 'INDEP' and 'DEP' columns.")
    
#     # Extract the required columns
#     indep = df['INDEP']
#     dep = df['DEP']

#     # Create a new Excel workbook and add a worksheet
#     workbook = xlsxwriter.Workbook(excel_file)
#     worksheet = workbook.add_worksheet()

#     # Write the data to the worksheet
#     worksheet.write('A1', 'DEP')
#     worksheet.write('B1', 'INDEP')

#     for i, (d, ind) in enumerate(zip(dep, indep), start=1):
#         worksheet.write(i, 0, d)
#         worksheet.write(i, 1, ind)

#     # Create a chart object
#     chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})

#     # Configure the chart
#     chart.add_series({
#         'name':       'INDEP vs DEP',
#         'categories': [worksheet.name, 1, 0, len(dep), 0],
#         'values':     [worksheet.name, 1, 1, len(indep), 1],
#         'marker':     {'type': 'circle', 'size': 5},
#         'trendline':  {'type': 'polynomial', 'order': 2, 'name': 'Polynomial Trendline'}
#     })
#     chart.add_series({
#         'name':       'INDEP vs DEP (Power Trendline)',
#         'categories': [worksheet.name, 1, 0, len(dep), 0],
#         'values':     [worksheet.name, 1, 1, len(indep), 1],
#         'marker':     {'type': 'circle', 'size': 5},
#         'trendline':  {'type': 'power', 'name': 'Power Trendline'}
#     })
#     chart.add_series({
#         'name':       'INDEP vs DEP (Exponential Trendline)',
#         'categories': [worksheet.name, 1, 0, len(dep), 0],
#         'values':     [worksheet.name, 1, 1, len(indep), 1],
#         'marker':     {'type': 'circle', 'size': 5},
#         'trendline':  {'type': 'exponential', 'name': 'Exponential Trendline'}
#     })

#     chart.set_title({'name': 'INDEP vs DEP with Trendlines'})
#     chart.set_x_axis({'name': 'DEP'})
#     chart.set_y_axis({'name': 'INDEP'})

#     # Insert the chart into the worksheet
#     worksheet.insert_chart('D2', chart)

#     # Close the workbook
#     workbook.close()

# def process_files(directory):
#     results = []
#     no_data_files = []

#     for filename in os.listdir(directory):
#         if filename.endswith(".csv"):
#             csv_path = os.path.join(directory, filename)
#             excel_path = os.path.join(directory, filename.replace('.csv', '.xlsx'))

#             try:
#                 plot_csv_to_excel(csv_path, excel_path)
#                 results.append((filename, "Success"))
#             except ValueError as e:
#                 print(f"Error processing {filename}: {e}")
#                 no_data_files.append(filename)
    
#     return pd.DataFrame(results, columns=['Filename', 'Status']), pd.DataFrame(no_data_files, columns=['Filename'])

# def main():
#     directory = '/Users/gabrielhernandez/Desktop/Test Code'  # Directory path
#     results_df, no_data_df = process_files(directory)
    
#     # Save results in the same directory
#     results_path = os.path.join(directory, 'trendline_results.csv')
#     no_data_path = os.path.join(directory, 'no_data_files.csv')
    
#     results_df.to_csv(results_path, index=False)
#     no_data_df.to_csv(no_data_path, index=False)
    
#     print(f"Processing complete. Results saved to '{results_path}' and '{no_data_path}'.")

# if __name__ == "__main__":
#     main()

# import pandas as pd
# import os
# import xlsxwriter

# def plot_csv_to_excel(csv_file, excel_file):
#     # Read the CSV file
#     df = pd.read_csv(csv_file)
    
#     # Check if necessary columns are present
#     if 'INDEP' not in df.columns or 'DEP' not in df.columns:
#         raise ValueError(f"CSV file {csv_file} must contain 'INDEP' and 'DEP' columns.")
    
#     # Extract the required columns
#     indep = df['INDEP']
#     dep = df['DEP']

#     # Create a new Excel workbook and add a worksheet
#     workbook = xlsxwriter.Workbook(excel_file)
#     worksheet = workbook.add_worksheet()

#     # Write the data to the worksheet
#     worksheet.write('A1', 'DEP')
#     worksheet.write('B1', 'INDEP')

#     for i, (d, ind) in enumerate(zip(dep, indep), start=1):
#         worksheet.write(i, 0, d)
#         worksheet.write(i, 1, ind)

#     # Create a chart object
#     chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})

#     # Configure the chart
#     chart.add_series({
#         'name':       'INDEP vs DEP',
#         'categories': [worksheet.name, 1, 0, len(dep), 0],
#         'values':     [worksheet.name, 1, 1, len(indep), 1],
#         'marker':     {'type': 'circle', 'size': 5},
#         'trendline':  {
#             'type': 'polynomial',
#             'order': 2,
#             'name': 'Polynomial Trendline',
#             'display_equation': True,
#             'display_r_squared': True
#         }
#     })
#     chart.add_series({
#         'name':       'INDEP vs DEP (Power Trendline)',
#         'categories': [worksheet.name, 1, 0, len(dep), 0],
#         'values':     [worksheet.name, 1, 1, len(indep), 1],
#         'marker':     {'type': 'circle', 'size': 5},
#         'trendline':  {
#             'type': 'power',
#             'name': 'Power Trendline',
#             'display_equation': True,
#             'display_r_squared': True
#         }
#     })
#     chart.add_series({
#         'name':       'INDEP vs DEP (Exponential Trendline)',
#         'categories': [worksheet.name, 1, 0, len(dep), 0],
#         'values':     [worksheet.name, 1, 1, len(indep), 1],
#         'marker':     {'type': 'circle', 'size': 5},
#         'trendline':  {
#             'type': 'exponential',
#             'name': 'Exponential Trendline',
#             'display_equation': True,
#             'display_r_squared': True
#         }
#     })

#     chart.set_title({'name': 'INDEP vs DEP with Trendlines'})
#     chart.set_x_axis({'name': 'DEP'})
#     chart.set_y_axis({'name': 'INDEP'})

#     # Insert the chart into the worksheet
#     worksheet.insert_chart('D2', chart)

#     # Close the workbook
#     workbook.close()

# def process_files(directory):
#     results = []
#     no_data_files = []

#     for filename in os.listdir(directory):
#         if filename.endswith(".csv"):
#             csv_path = os.path.join(directory, filename)
#             excel_path = os.path.join(directory, filename.replace('.csv', '.xlsx'))

#             try:
#                 plot_csv_to_excel(csv_path, excel_path)
#                 results.append((filename, "Success"))
#             except ValueError as e:
#                 print(f"Error processing {filename}: {e}")
#                 no_data_files.append(filename)
    
#     return pd.DataFrame(results, columns=['Filename', 'Status']), pd.DataFrame(no_data_files, columns=['Filename'])

# def main():
#     directory = '/Users/gabrielhernandez/Desktop/Test Code'  # Directory path
#     results_df, no_data_df = process_files(directory)
    
#     # Save results in the same directory
#     results_path = os.path.join(directory, 'trendline_results.csv')
#     no_data_path = os.path.join(directory, 'no_data_files.csv')
    
#     results_df.to_csv(results_path, index=False)
#     no_data_df.to_csv(no_data_path, index=False)
    
#     print(f"Processing complete. Results saved to '{results_path}' and '{no_data_path}'.")

# if __name__ == "__main__":
#     main()

# import pandas as pd
# import os
# import xlsxwriter

# def plot_csv_to_excel(csv_file, excel_file):
#     # Read the CSV file
#     df = pd.read_csv(csv_file)
    
#     # Check if necessary columns are present
#     if 'INDEP' not in df.columns or 'DEP' not in df.columns:
#         raise ValueError(f"CSV file {csv_file} must contain 'INDEP' and 'DEP' columns.")
    
#     # Extract the required columns
#     indep = df['INDEP']
#     dep = df['DEP']

#     # Create a new Excel workbook and add a worksheet
#     workbook = xlsxwriter.Workbook(excel_file)
#     worksheet = workbook.add_worksheet()

#     # Write the data to the worksheet
#     worksheet.write('A1', 'DEP')
#     worksheet.write('B1', 'INDEP')

#     for i, (d, ind) in enumerate(zip(dep, indep), start=1):
#         worksheet.write(i, 0, d)
#         worksheet.write(i, 1, ind)

#     # Create a chart object
#     chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})

#     # Configure the chart
#     chart.add_series({
#         'name':       'INDEP vs DEP',
#         'categories': [worksheet.name, 1, 0, len(dep), 0],
#         'values':     [worksheet.name, 1, 1, len(indep), 1],
#         'marker':     {'type': 'circle', 'size': 5},
#         'trendline':  {
#             'type': 'polynomial',
#             'order': 2,
#             'name': 'Polynomial Trendline',
#             'display_equation': True,
#             'display_r_squared': True
#         }
#     })
#     chart.add_series({
#         'name':       'INDEP vs DEP (Power Trendline)',
#         'categories': [worksheet.name, 1, 0, len(dep), 0],
#         'values':     [worksheet.name, 1, 1, len(indep), 1],
#         'marker':     {'type': 'circle', 'size': 5},
#         'trendline':  {
#             'type': 'power',
#             'name': 'Power Trendline',
#             'display_equation': True,
#             'display_r_squared': True
#         }
#     })
#     chart.add_series({
#         'name':       'INDEP vs DEP (Exponential Trendline)',
#         'categories': [worksheet.name, 1, 0, len(dep), 0],
#         'values':     [worksheet.name, 1, 1, len(indep), 1],
#         'marker':     {'type': 'circle', 'size': 5},
#         'trendline':  {
#             'type': 'exponential',
#             'name': 'Exponential Trendline',
#             'display_equation': True,
#             'display_r_squared': True
#         }
#     })

#     chart.set_title({'name': 'INDEP vs DEP with Trendlines'})
#     chart.set_x_axis({'name': 'DEP'})
#     chart.set_y_axis({'name': 'INDEP'})

#     # Insert the chart into the worksheet
#     worksheet.insert_chart('D2', chart)

#     # Close the workbook
#     workbook.close()

# def process_files(directory):
#     results = []
#     no_data_files = []

#     for filename in os.listdir(directory):
#         if filename.endswith(".csv"):
#             csv_path = os.path.join(directory, filename)
#             excel_path = os.path.join(directory, filename.replace('.csv', '.xlsx'))

#             try:
#                 plot_csv_to_excel(csv_path, excel_path)
#                 results.append((filename, "Success"))
#             except ValueError as e:
#                 print(f"Error processing {filename}: {e}")
#                 no_data_files.append(filename)
    
#     return pd.DataFrame(results, columns=['Filename', 'Status']), pd.DataFrame(no_data_files, columns=['Filename'])

# def main():
#     directory = '/Users/gabrielhernandez/Desktop/lildata'  # Directory path
#     results_df, no_data_df = process_files(directory)
    
#     # Save results in the same directory
#     results_path = os.path.join(directory, 'trendline_results.csv')
#     no_data_path = os.path.join(directory, 'no_data_files.csv')
    
#     results_df.to_csv(results_path, index=False)
#     no_data_df.to_csv(no_data_path, index=False)
    
#     print(f"Processing complete. Results saved to '{results_path}' and '{no_data_path}'.")

# if __name__ == "__main__":
#     main()
