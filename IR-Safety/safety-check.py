"""
Eye Safety Analysis for IR Illuminator - IEC 62471 Compliance Check

This script evaluates the safety of an infrared illuminator for eye exposure according
to the IEC 62471 standard "Photobiological safety of lamps and lamp systems".

The analysis covers two main safety criteria:
1. Infrared radiation hazard (Section 4.3.7): Total irradiance limit for thermal damage
2. Retinal burn hazard (Section 4.3.6): Weighted spectral radiance limit for retinal damage

Methodology:
- Power measurements taken with calibrated Thorlabs S121C sensor
- Spectral data interpolated from manufacturer specifications
- Sensor responsivity correction applied for accurate spectral irradiance calculation
- Safety limits evaluated per IEC 62471 formulas

Equipment:
- Illuminator: IR LED (M940L3) with known relative spectral intensity
- Power sensor: Thorlabs S121C (calibrated at 940nm)
- Measurement distance: 0.5m from illuminator

Author: Ryan Ressmeyer
Date: 10/8/2025
Standard: IEC 62471:2006 (Photobiological safety of lamps and lamp systems)
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from pathlib import Path

# Configure matplotlib to use TkAgg backend for interactive plots
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for GUI, or 'Agg' for non-interactive
import matplotlib.pyplot as plt

# Get the project directory from the script location
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_header(text, level=1):
    """
    Print a formatted header for terminal output.

    Args:
        text (str): Header text
        level (int): Header level (1 for main, 2 for subsection)
    """
    if level == 1:
        print("\n" + "="*80)
        print(f"  {text}")
        print("="*80)
    else:
        print(f"\n{text}")
        print("-"*len(text))


# =============================================================================
# MEASUREMENT DATA AND CONSTANTS
# =============================================================================
# Power measurements taken with Thorlabs S121C sensor at 0.5m distance
# These represent the total optical power incident on the sensor active area

# Measured optical power values
eye_power = 3.5e-3   # W - Power measured when illuminator positioned for eye illumination
face_power = 8.5e-3  # W - Power measured when illuminator positioned for face illumination

# For this analysis, we focus on the eye illumination scenario as it represents
# the highest risk exposure condition for eye safety evaluation
measured_power = eye_power

print_header("ILLUMINATOR SAFETY ANALYSIS", level=1)
print(f"Measured power (eye position):  {eye_power*1000:.1f} mW")
print(f"Measured power (face position): {face_power*1000:.1f} mW")
print(f"Analysis will use eye position data: {measured_power*1000:.1f} mW")

# =============================================================================
# ILLUMINATOR SPECTRAL CHARACTERISTICS
# =============================================================================
# Load the relative spectral intensity of the Thorlabs M940L3 illuminator
# This data describes how the illuminator's output varies across wavelengths

print_header("LOADING ILLUMINATOR SPECTRAL DATA", level=1)

illuminator_df = pd.read_csv(SCRIPT_DIR / 'M940L3_Relative_Spectral_Intensity.csv')
illuminator_wavelengths = illuminator_df['Wavelength (nm)'].to_numpy()
illuminator_relative_intensity = illuminator_df['Relative Intensity'].to_numpy()

# Create interpolation function for smooth spectral calculations
illuminator_interp = CubicSpline(illuminator_wavelengths, illuminator_relative_intensity)

print(f"Loaded spectral data:  {len(illuminator_wavelengths)} data points")
print(f"Wavelength range:      {np.min(illuminator_wavelengths):.0f} - {np.max(illuminator_wavelengths):.0f} nm")
print(f"Peak wavelength:       ~{illuminator_wavelengths[np.argmax(illuminator_relative_intensity)]:.0f} nm")

# Plot the illuminator spectral characteristics
wavelength_plot_range = np.arange(np.min(illuminator_wavelengths),
                                 np.max(illuminator_wavelengths), 1)
plt.figure(figsize=(10, 6))
plt.plot(wavelength_plot_range, illuminator_interp(wavelength_plot_range), linewidth=2)
#plt.scatter(illuminator_wavelengths, illuminator_relative_intensity, color='red', s=30, alpha=0.7,
           #label='Data points')
plt.legend()
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Relative Intensity', fontsize=12)
plt.title('Illuminator Relative Spectral Intensity', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# SENSOR SPECTRAL RESPONSIVITY
# =============================================================================
# Load the spectral responsivity of the Thorlabs S121C sensor
# This correction is crucial because the sensor's sensitivity varies with wavelength

print_header("LOADING SENSOR RESPONSIVITY DATA", level=1)

sensor_df = pd.read_csv(SCRIPT_DIR / 'S121C_Responsivity.csv')
sensor_wavelengths = sensor_df['Wavelength (nm)'].to_numpy()
sensor_responsivity = sensor_df['Responsivity (mA/W)'].to_numpy()

# Create interpolation function for sensor correction
sensor_interp = CubicSpline(sensor_wavelengths, sensor_responsivity)

print(f"Loaded sensor data:        {len(sensor_wavelengths)} data points")
print(f"Responsivity range:        {np.min(sensor_responsivity):.0f} - {np.max(sensor_responsivity):.0f} mA/W")
print(f"Sensor wavelength range:   {np.min(sensor_wavelengths):.0f} - {np.max(sensor_wavelengths):.0f} nm")

# =============================================================================
# SENSOR CALIBRATION AND SPECTRAL CORRECTION
# =============================================================================
# The Thorlabs S121C sensor is calibrated for 940nm light, but our illuminator
# has a broad spectrum. We need to correct for the wavelength-dependent sensitivity.

print_header("SENSOR CALIBRATION ANALYSIS", level=1)

# Define wavelength grid for calculations
w0 = np.min(illuminator_wavelengths)
w1 = np.max(illuminator_wavelengths)
dw = 5  # nm - wavelength step size for numerical integration

wavelengths = np.arange(w0, w1, dw)

# Sensor specifications
calibrated_wavelength = 940  # nm - wavelength at which sensor is calibrated
sensor_diameter = 9.7e-3     # m - active area diameter
sensor_area = np.pi * (sensor_diameter/2)**2  # m^2 - active area

print(f"Wavelength range for analysis: {w0:.0f} - {w1:.0f} nm")
print(f"Wavelength step size:          {dw} nm")
print(f"Sensor calibrated at:          {calibrated_wavelength} nm")
print(f"Sensor active area:            {sensor_area*1e6:.2f} mm²")

# Calculate normalized sensor sensitivity relative to calibration wavelength
sensor_sensitivity = sensor_interp(wavelengths) / sensor_interp(calibrated_wavelength)

# Plot sensor responsivity and normalized sensitivity side by side
sensor_plot_range = np.arange(np.min(sensor_wavelengths), np.max(sensor_wavelengths), 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left subplot: Spectral responsivity
ax1.plot(sensor_plot_range, sensor_interp(sensor_plot_range), linewidth=2)
#ax1.scatter(sensor_wavelengths, sensor_responsivity, color='red', s=30, alpha=0.7,
           #label='Data points')
ax1.set_xlabel('Wavelength (nm)', fontsize=12)
ax1.set_ylabel('Responsivity (mA/W)', fontsize=12)
ax1.set_title('Thorlabs S121C Sensor Spectral Responsivity', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right subplot: Normalized sensitivity
ax2.plot(wavelengths, sensor_sensitivity, linewidth=2, label='Normalized Sensitivity')
ax2.plot([calibrated_wavelength, calibrated_wavelength], [0, 1],
         color='red', linestyle='--',
         label=f'Calibrated Wavelength ({calibrated_wavelength} nm)')
ax2.set_ylim(0, 1.1)
ax2.set_xlabel('Wavelength (nm)', fontsize=12)
ax2.set_ylabel('Normalized Sensitivity', fontsize=12)
ax2.set_title('Sensor Normalized Spectral Sensitivity', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'sensor_responsivity.jpg', dpi=300)
plt.show()

# =============================================================================
# MATHEMATICAL DERIVATION: SPECTRAL IRRADIANCE CALCULATION
# =============================================================================
# To calculate absolute spectral irradiance from relative measurements,
# we need to solve for the scaling factor that matches our power measurement.

print_header("MATHEMATICAL DERIVATION", level=1)
print("Converting relative spectral intensity to absolute spectral irradiance...\n")

# The fundamental equation relating measured power to spectral irradiance
print("Equation: P = ∫ E(λ) · S(λ) · A dλ")
print("  Power = Integral of [Spectral Irradiance × Sensor Responsivity × Area] over wavelength\n")

print("Where:")
print("  P    = Measured power (W)")
print("  E(λ) = Spectral irradiance (W/m²/nm)")
print("  S(λ) = Sensor responsivity (A/W)")
print("  A    = Sensor area (m²)\n")

# We only know relative spectral irradiance, so we introduce a scaling factor
print("Equation: E(λ) = I · E_rel(λ)")
print("  Spectral Irradiance = Scaling Factor × Relative Spectral Intensity\n")

print("Where:")
print("  I       = Scaling factor (W/m²)")
print("  E_rel(λ) = Relative spectral intensity (dimensionless)\n")

# Substituting into the power equation
print("Equation: P = ∫ I · E_rel(λ) · S(λ) · A dλ")
print("  Power = Scaling Factor × Integral of [Relative Intensity × Responsivity × Area]\n")

# Rearranging to solve for the scaling factor
print("Equation: I = P / [∫ E_rel(λ) · S(λ) · A dλ]")
print("  Scaling Factor = Measured Power / [Integral of Relative Intensity × Responsivity × Area]\n")

# For numerical implementation using discrete summation
print("Equation: I = P / [Σ E_rel(λ) · S(λ) · Δλ · A]")
print("  Scaling Factor = Measured Power / [Sum of Relative Intensity × Responsivity × Wavelength Step × Area]")

# =============================================================================
# SPECTRAL IRRADIANCE CALCULATION
# =============================================================================
# Calculate the scaling factor and absolute spectral irradiance

print_header("CALCULATING ABSOLUTE SPECTRAL IRRADIANCE", level=1)

# Calculate the denominator (integral term) using numerical integration
denominator = np.dot(illuminator_interp(wavelengths), sensor_sensitivity) * dw * sensor_area

# Calculate scaling factor I
scaling_factor_I = measured_power / denominator

print(f"Measured power:              {measured_power*1000:.1f} mW")
print(f"Integral term:               {denominator:.2e}")
print(f"Scaling factor I:            {scaling_factor_I:.2f} W/m²")

# Calculate absolute spectral irradiance
illuminator_spectral_irradiance = scaling_factor_I * illuminator_interp(wavelengths)  # W/m²/nm

# Calculate total irradiance by integrating spectral irradiance
total_calculated_irradiance = np.sum(illuminator_spectral_irradiance) * dw  # W/m²

print(f"Peak spectral irradiance:    {np.max(illuminator_spectral_irradiance):.2f} W/m²/nm")
print(f"Total calculated irradiance: {total_calculated_irradiance:.2f} W/m²")

# Plot the calculated spectral irradiance
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, illuminator_spectral_irradiance, linewidth=2)
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Spectral Irradiance (W/m²/nm)', fontsize=12)
plt.title('Illuminator Absolute Spectral Irradiance', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'spectral_irradiance.jpg', dpi=300)
plt.show()

# =============================================================================
# IEC 62471 SAFETY EVALUATION - INFRARED RADIATION HAZARD (Section 4.3.7)
# =============================================================================
# Evaluate thermal hazard from infrared radiation for indefinite exposure

print_header("IEC 62471 INFRARED RADIATION HAZARD ASSESSMENT", level=1)
print("Standard: IEC 62471:2006, Section 4.3.7\n")

# Display the safety criterion equation
print("Equation: E_IR = Σ(λ=700nm to 3000nm) E_λ · Δλ < 100 W/m²")
print("  IR Irradiance = Sum of spectral irradiance from 700-3000nm < 100 W/m²\n")

print("For indefinite exposure (continuous viewing), total IR irradiance must be < 100 W/m²\n")

# For time-limited exposures, display the time-dependent limit
print("Equation: E_IR = Σ(λ=700nm to 3000nm) E_λ · Δλ ≤ 18000 · t^(-0.75) W/m²")
print("  For exposure time t (seconds): IR Irradiance ≤ 18000 × t^(-0.75) W/m²")
print("  Where t is the exposure time in seconds")

# Calculate total IR irradiance in the specified wavelength range
irradiance_limit_indefinite = 100  # W/m² for indefinite exposure
ir_wavelength_mask = (wavelengths >= 700) & (wavelengths <= 3000)
total_ir_irradiance = np.sum(illuminator_spectral_irradiance[ir_wavelength_mask]) * dw

# Safety evaluation
print(f"\nCalculated total IR irradiance (700-3000 nm): {total_ir_irradiance:.2f} W/m²")
print(f"Safety limit for indefinite exposure:         {irradiance_limit_indefinite} W/m²")

if total_ir_irradiance < irradiance_limit_indefinite:
    safety_margin = irradiance_limit_indefinite / total_ir_irradiance
    print("\n" + "="*80)
    print("✅ SAFE: Illuminator meets IEC 62471 infrared radiation safety requirements")
    print(f"   Irradiance ({total_ir_irradiance:.2f} W/m²) < Limit ({irradiance_limit_indefinite} W/m²)")
    print(f"   Safety margin: {safety_margin:.1f}×")
    print("="*80)
else:
    # Calculate maximum safe exposure time
    maximum_exposure_time = (total_ir_irradiance / 18000)**(-4/3)
    print("\n" + "="*80)
    print("⚠️  HAZARDOUS: Illuminator exceeds IEC 62471 infrared radiation safety limits")
    print(f"   Irradiance ({total_ir_irradiance:.2f} W/m²) > Limit ({irradiance_limit_indefinite} W/m²)")
    print(f"   Maximum safe exposure time: {maximum_exposure_time:.2f} seconds ({maximum_exposure_time/60:.2f} minutes)")
    print("="*80)


# =============================================================================
# IEC 62471 SAFETY EVALUATION - RETINAL BURN HAZARD (Section 4.3.6)
# =============================================================================
# Evaluate retinal thermal damage hazard using weighted spectral radiance

print_header("IEC 62471 RETINAL BURN HAZARD ASSESSMENT", level=1)
print("Standard: IEC 62471:2006, Section 4.3.6\n")

# Display the retinal burn hazard criterion
print("Equation: L_IR = Σ(λ=780nm to 1400nm) L_λ · R(λ) · Δλ < 6000/α W/m²/sr")
print("  Weighted IR Radiance = Sum of [Radiance × Weighting Function] from 780-1400nm < 6000/α W/m²/sr\n")

print("Where:")
print("  L_λ  = Spectral radiance (W/m²/sr/nm)")
print("  R(λ) = Retinal burn hazard weighting function (Table 4.2)")
print("  Δλ   = Wavelength step (nm)")
print("  α    = Angular subtense of the source (radians)\n")

print("The retinal burn hazard assessment accounts for:")
print("  • Wavelength-dependent absorption in retinal tissue")
print("  • Angular size of the source (smaller sources are more dangerous)")
print("  • Spectral weighting based on known damage thresholds")
# =============================================================================
# RETINAL BURN HAZARD WEIGHTING FUNCTION (IEC 62471 Table 4.2)
# =============================================================================

def retinal_burn_hazard_function(wavelength_nm):
    """
    Calculates the IEC 62471 retinal burn hazard weighting function R(λ).

    This function covers the spectral range from 380 nm to 1400 nm as specified
    in Table 4.2 of the standard. It uses linear interpolation for the range
    380 nm to 500 nm where discrete values are provided.

    Args:
        wavelength_nm (float or np.ndarray): The wavelength(s) in nanometers.

    Returns:
        float or np.ndarray: The corresponding R(λ) weighting factor(s).
    """
    # Ensure input is a numpy array for vectorized operations
    wavelength_nm = np.asarray(wavelength_nm)
    
    # Initialize output array with zeros
    r_lambda = np.zeros_like(wavelength_nm, dtype=float)

    # --- Range 1: 380 nm to 500 nm (Interpolation from table data) ---
    # Data points from IEC 62471, Table 4.2 for R(λ)
    wavelength_points = np.array([
        380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440,
        445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495, 500
    ])
    r_values = np.array([
        0.1, 0.13, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 9.0, 9.5, 9.8, 10.0, 10.0,
        9.7, 9.4, 9.0, 8.0, 7.0, 6.2, 5.5, 4.5, 4.0, 2.2, 1.6, 1.0
    ])
    
    # Define the condition for this range
    cond1 = (wavelength_nm >= 380) & (wavelength_nm < 500)
    r_lambda[cond1] = np.interp(wavelength_nm[cond1], wavelength_points, r_values)

    # --- Range 2: 500 nm to 700 nm ---
    cond2 = (wavelength_nm >= 500) & (wavelength_nm <= 700)
    r_lambda[cond2] = 1.0

    # --- Range 3: 700 nm to 1050 nm ---
    cond3 = (wavelength_nm > 700) & (wavelength_nm <= 1050)
    r_lambda[cond3] = 10**((700 - wavelength_nm[cond3]) / 500)

    # --- Range 4: 1050 nm to 1150 nm ---
    cond4 = (wavelength_nm > 1050) & (wavelength_nm <= 1150)
    r_lambda[cond4] = 0.2

    # --- Range 5: 1150 nm to 1200 nm ---
    cond5 = (wavelength_nm > 1150) & (wavelength_nm <= 1200)
    r_lambda[cond5] = 0.2 * 10**(0.02 * (1150 - wavelength_nm[cond5]))
    
    # --- Range 6: 1200 nm to 1400 nm ---
    cond6 = (wavelength_nm > 1200) & (wavelength_nm <= 1400)
    r_lambda[cond6] = 0.02

    # If the input was a single float, return a float
    if isinstance(wavelength_nm, (int, float)):
        return r_lambda.item()
    return r_lambda

# Define the wavelength range for plotting
r_wavelengths = np.arange(380, 1401, 1)

# Get the R(λ) values
r_lambda_values = retinal_burn_hazard_function(r_wavelengths)

    
# =============================================================================
# RETINAL HAZARD CALCULATION
# =============================================================================
# Calculate illuminator geometry and angular subtense for retinal burn assessment

print_header("ILLUMINATOR GEOMETRY ANALYSIS", level=1)

# Illuminator geometric parameters
illuminator_distance = 0.5      # m - measurement distance
illuminator_diameter = 0.05     # m - illuminator diameter
illuminator_angle = illuminator_diameter / illuminator_distance  # rad (small angle approximation)
# Note: minimum angle for point source is 0.011 rad (11 mrad) as per IEC 62471
# For long exposures this angle scales to 0.1 rad
if illuminator_angle < 0.011:
    print(f"Angular subtense ({illuminator_angle:.3f} rad) is less than minimum; using 0.011 rad")
    illuminator_angle = 0.011  # rad
    illuminator_diameter = illuminator_angle * illuminator_distance  # m
    print(f"Adjusted illuminator diameter: {illuminator_diameter*1000:.2f} mm")

# Calculate solid angle subtended by illuminator
illuminator_solid_angle = 2*np.pi*(1 - illuminator_distance/np.sqrt(illuminator_distance**2 + (illuminator_diameter/2)**2))  # sr

# Angular subtense (α) used in retinal burn hazard calculation
alpha = illuminator_angle

print(f"Illuminator distance:   {illuminator_distance} m")
print(f"Illuminator diameter:   {illuminator_diameter*1000:.0f} mm")
print(f"Angular subtense (α):   {illuminator_angle*1000:.2f} mrad")
print(f"Solid angle:            {illuminator_solid_angle:.4f} sr")

# Convert spectral irradiance to spectral radiance
illuminator_spectral_radiance = illuminator_spectral_irradiance / illuminator_solid_angle  # W/m²/sr/nm

# Plot spectral radiance and retinal burn hazard weighting function in subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left subplot: Spectral radiance
ax1.plot(wavelengths, illuminator_spectral_radiance, label='Spectral Radiance', linewidth=2)
ax1.set_title('Illuminator Spectral Radiance', fontsize=14)
ax1.set_xlabel('Wavelength (nm)', fontsize=12)
ax1.set_ylabel('Spectral Radiance (W/m²/sr/nm)', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(wavelengths[0], wavelengths[-1])

# Right subplot: Retinal burn hazard weighting function
ax2.plot(r_wavelengths, r_lambda_values, label='R(λ) - Retinal Burn Hazard', color='firebrick', linewidth=2)
# Use a logarithmic scale for the y-axis to better visualize the variations
ax2.set_yscale('log')
ax2.set_title('IEC 62471: Retinal Burn Hazard Weighting Function R(λ)', fontsize=14)
ax2.set_xlabel('Wavelength (nm)', fontsize=12)
ax2.set_ylabel('Spectral Weighting R(λ) (Log Scale)', fontsize=12)
ax2.legend(fontsize=10)
ax2.set_xlim(380, 1400)
ax2.grid(True, which="both", ls="--", c='0.65')

plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'spectral_radiance_hazard_function.jpg', dpi=300)
plt.show()
# Calculate retinal burn hazard limit
retinal_hazard_limit = 6000 / alpha  # W/m²/sr

print(f"Retinal burn hazard limit: {retinal_hazard_limit:.0f} W/m²/sr")

# Calculate weighted retinal exposure
retinal_wavelength_mask = (wavelengths >= 780) & (wavelengths <= 1400)
retinal_wavelengths = wavelengths[retinal_wavelength_mask]
retinal_radiance = illuminator_spectral_radiance[retinal_wavelength_mask]
retinal_weighting = retinal_burn_hazard_function(retinal_wavelengths)

weighted_retinal_exposure = np.sum(retinal_radiance * retinal_weighting) * dw  # W/m²/sr

print(f"\nCalculated weighted retinal exposure: {weighted_retinal_exposure:.2f} W/m²/sr")
print(f"Safety limit:                         {retinal_hazard_limit:.0f} W/m²/sr")

# Final retinal burn hazard assessment
if weighted_retinal_exposure < retinal_hazard_limit:
    retinal_safety_margin = retinal_hazard_limit / weighted_retinal_exposure
    print("\n" + "="*80)
    print("✅ SAFE: Illuminator meets IEC 62471 retinal burn hazard safety requirements")
    print(f"   Weighted exposure ({weighted_retinal_exposure:.2f} W/m²/sr) < Limit ({retinal_hazard_limit:.0f} W/m²/sr)")
    print(f"   Safety margin: {retinal_safety_margin:.1f}×")
    print("="*80)
else:
    print("\n" + "="*80)
    print("⚠️  HAZARDOUS: Illuminator exceeds IEC 62471 retinal burn hazard safety limits")
    print(f"   Weighted exposure ({weighted_retinal_exposure:.2f} W/m²/sr) > Limit ({retinal_hazard_limit:.0f} W/m²/sr)")
    print("="*80)


