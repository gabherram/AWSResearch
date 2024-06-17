import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Function to fit different models and calculate R-squared
def calculate_trendlines(x, y):
    results = {}

    # Power Trendline (y = a * x^b)
    def power_law(x, a, b):
        return a * np.power(x, b)

    try:
        popt, _ = curve_fit(power_law, x, y, maxfev=10000)
        y_pred = power_law(x, *popt)
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        results['Power'] = (r_squared, popt)
    except Exception as e:
        print(f"Power model fitting failed: {e}")
        results['Power'] = (None, None)

    # Exponential Trendline (y = a * exp(b * x))
    def exponential(x, a, b):
        return a * np.exp(b * x)

    try:
        popt, _ = curve_fit(exponential, x, y, maxfev=10000)
        y_pred = exponential(x, *popt)
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        results['Exponential'] = (r_squared, popt)
    except Exception as e:
        print(f"Exponential model fitting failed: {e}")
        results['Exponential'] = (None, None)

    # Polynomial Trendline (y = a * x^2 + b * x + c)
    try:
        poly_coeffs = np.polyfit(x, y, 2)
        poly_eq = np.poly1d(poly_coeffs)
        y_pred = poly_eq(x)
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        results['Polynomial'] = (r_squared, poly_coeffs)
    except Exception as e:
        print(f"Polynomial model fitting failed: {e}")
        results['Polynomial'] = (None, None)

    return results

# Main processing function
def process_file(file_path):
    results = []
    no_data_sites = []

    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Check if required columns are present
        if 'siteID' not in df.columns or 'stage' not in df.columns or 'flow' not in df.columns:
            raise ValueError("Required columns not found in the file")

        # Group data by siteID
        grouped = df.groupby('siteID')
        
        for site_id, group in grouped:
            indep = group['stage'].dropna().astype(float).values
            dep = group['flow'].dropna().astype(float).values

            if len(indep) == 0 or len(dep) == 0:
                no_data_sites.append((site_id, "No data available"))
                continue

            trendline_results = calculate_trendlines(indep, dep)
            if not trendline_results:
                no_data_sites.append((site_id, "No trendline results"))
                continue

            best_fit = max(trendline_results, key=lambda k: trendline_results[k][0] if trendline_results[k][0] is not None else -1)
            best_r_squared = trendline_results[best_fit][0]
            best_coeffs = trendline_results[best_fit][1]

            if best_coeffs is not None and len(best_coeffs) >= 2:
                variable_a, variable_b = best_coeffs[0], best_coeffs[1]
            else:
                variable_a, variable_b = None, None

            results.append([site_id, trendline_results['Power'][0], trendline_results['Exponential'][0], trendline_results['Polynomial'][0], best_fit, best_r_squared, best_coeffs, variable_a, variable_b])
            print(f"Processed {site_id} with best fit: {best_fit} and R-squared: {best_r_squared}")

    except Exception as e:
        print(f"Failed to process file: {e}")

    results_df = pd.DataFrame(results, columns=['Site ID', 'Power R-squared', 'Exponential R-squared', 'Polynomial R-squared', 'Best Fit', 'Best R-squared', 'Coefficients', 'Variable a', 'Variable b'])
    no_data_df = pd.DataFrame(no_data_sites, columns=['Site ID', 'Message'])

    return results_df, no_data_df

# Run the processing function and save results
def main():
    file_path = '/Users/gabrielhernandez/Desktop/Data-AWS RESEARCH/Rating_curve.xlsx'
    results_df, no_data_df = process_file(file_path)
    
    # Save results in the same directory
    directory = os.path.dirname(file_path)
    results_path = os.path.join(directory, 'trendline_results.xlsx')
    no_data_path = os.path.join(directory, 'no_data_sites.xlsx')
    
    results_df.to_excel(results_path, index=False)
    no_data_df.to_excel(no_data_path, index=False)
    
    print(f"Processing complete. Results saved to '{results_path}' and '{no_data_path}'.")

if __name__ == "__main__":
    main()
