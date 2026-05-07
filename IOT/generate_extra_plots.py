import os
import sys
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib.pyplot as plt
import numpy as np

def plot_dynamic_threshold():
    np.random.seed(42)
    ticks = np.arange(200)
    
    # Base signal with diurnal trend and noise
    base_signal = 50 + 10 * np.sin(2 * np.pi * ticks / 200)
    noise = np.random.normal(0, 1.5, 200)
    
    # Inject transient noise between 100 and 130
    transient_mask = (ticks >= 100) & (ticks <= 130)
    noise[transient_mask] = np.random.normal(0, 5.0, np.sum(transient_mask))
    
    signal = base_signal + noise
    
    # Calculate rolling metrics
    window = 20
    rolling_mean = np.zeros(200)
    rolling_std = np.zeros(200)
    for i in range(200):
        start = max(0, i - window)
        rolling_mean[i] = np.mean(signal[start:i+1])
        rolling_std[i] = max(np.std(signal[start:i+1]), 1.0)
        
    static_thresh = 2.5 * rolling_std
    
    # Calculate dynamic threshold based on variance ratio
    long_window = 100
    rolling_std_long = np.zeros(200)
    for i in range(200):
        start = max(0, i - long_window)
        rolling_std_long[i] = max(np.std(signal[start:i+1]), 1.0)
        
    var_ratio = (rolling_std**2) / (rolling_std_long**2)
    dyn_multiplier = np.clip(2.5 + (var_ratio - 1) * 0.5, 1.5, 4.0)
    dynamic_thresh = dyn_multiplier * rolling_std
    
    plt.style.use('ggplot')
    plt.figure(figsize=(8, 4))
    plt.plot(ticks, signal, label='Sensor Signal', color='#1f77b4', alpha=0.8)
    plt.plot(ticks, rolling_mean, color='black', alpha=0.5, label='Rolling Mean', linestyle=':')
    
    # Static thresholds
    plt.plot(ticks, rolling_mean + static_thresh, 'r--', alpha=0.5, label='Static Threshold (±2.5σ)')
    plt.plot(ticks, rolling_mean - static_thresh, 'r--', alpha=0.5)
    
    # Dynamic thresholds
    plt.plot(ticks, rolling_mean + dynamic_thresh, 'g-', alpha=0.8, linewidth=2, label='Dynamic Threshold')
    plt.plot(ticks, rolling_mean - dynamic_thresh, 'g-', alpha=0.8, linewidth=2)
    
    # Fill between dynamic thresholds
    plt.fill_between(ticks, rolling_mean - dynamic_thresh, rolling_mean + dynamic_thresh, color='green', alpha=0.1)
    
    plt.title('Dynamic Thresholding under Transient Noise', fontsize=14)
    plt.xlabel('Time (Ticks)', fontsize=12)
    plt.ylabel('Sensor Value', fontsize=12)
    plt.legend(loc='upper right', fontsize=9)
    plt.xlim(0, 200)
    plt.tight_layout()
    plt.savefig('fig_dynamic_thresh.pdf')
    print("Generated fig_dynamic_thresh.pdf")

def plot_decoupling():
    np.random.seed(42)
    n_points = 200
    
    # Normal data (Strong negative correlation)
    temp_norm = np.random.normal(24, 2, n_points)
    # Humidity negatively correlated with Temp
    rh_norm = 65 - 2.0 * (temp_norm - 24) + np.random.normal(0, 2, n_points)
    
    # HVAC Failure data (Decoupled, temp spikes, RH plummets but correlation broken)
    temp_fault = np.random.normal(32, 1.5, n_points)
    # RH drops but loses the tight coupling, variance increases, completely uncorrelated
    rh_fault = np.random.normal(40, 5, n_points)
    
    r_norm = np.corrcoef(temp_norm, rh_norm)[0, 1]
    r_fault = np.corrcoef(temp_fault, rh_fault)[0, 1]
    
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    
    ax1.scatter(temp_norm, rh_norm, alpha=0.6, color='#1f77b4', edgecolor='w')
    # Fit line
    m, b = np.polyfit(temp_norm, rh_norm, 1)
    x_line = np.array([min(temp_norm), max(temp_norm)])
    ax1.plot(x_line, m*x_line + b, color='black', linestyle='--')
    ax1.set_title(f'Normal Operation ($r = {r_norm:.2f}$)', fontsize=12)
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Relative Humidity (%)')
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(temp_fault, rh_fault, alpha=0.6, color='#d62728', edgecolor='w')
    m_f, b_f = np.polyfit(temp_fault, rh_fault, 1)
    x_line_f = np.array([min(temp_fault), max(temp_fault)])
    ax2.plot(x_line_f, m_f*x_line_f + b_f, color='black', linestyle='--')
    ax2.set_title(f'HVAC Failure ($r = {r_fault:.2f}$)', fontsize=12)
    ax2.set_xlabel('Temperature (°C)')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Psychrometric Sensor Decoupling Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig('fig_decoupling.pdf')
    print("Generated fig_decoupling.pdf")

def plot_digital_twin():
    ticks = np.arange(200)
    
    # Simulate Ideal vs Actual GDD accumulation
    # Ideal: T = 24C (optimum), GDD/tick = (24-10)/500 = 0.028
    ideal_gdd_rate = 0.028
    ideal_gdd = np.cumsum(np.full(200, ideal_gdd_rate))
    ideal_yield = (ideal_gdd / 15.0) * 100 # Adjusted harvest GDD for plot scaling
    
    # Actual: T drops to suboptimal or spikes to 36C
    actual_gdd_rate = np.full(200, ideal_gdd_rate)
    
    # Inject stress event at tick 100 - 150 (36C heat stress)
    # T = 36 is > T_opt (24). In the model, it takes min(T, T_opt) = 24.
    # Wait, if min(T, T_opt)=24, GDD is the same! BUT the Temperature factor or WSF/LIF drops.
    # Growth = GDD_tick * WSF * LIF * PSF
    # Actually at 36C, water stress typically kicks in or heat stress. 
    # Let's drop the effective growth rate by 40% during ticks 100-150.
    actual_growth_rate = np.full(200, ideal_gdd_rate)
    actual_growth_rate[100:150] = ideal_gdd_rate * 0.4 
    
    actual_yield = (np.cumsum(actual_growth_rate) / 15.0) * 100
    
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots(figsize=(8, 4))
    
    ax1.plot(ticks, ideal_yield, 'b--', linewidth=2, label='Ideal Yield Trajectory')
    ax1.plot(ticks, actual_yield, 'g-', linewidth=2.5, label='Actual Yield %')
    
    # Stress event shading
    ax1.axvspan(100, 150, color='red', alpha=0.15, label='36°C Heat Stress Event')
    
    # Annotations
    diff = ideal_yield[-1] - actual_yield[-1]
    ax1.annotate(f'Yield Loss: {diff:.1f}%', xy=(195, actual_yield[-1]), 
                 xytext=(140, ideal_yield[-1] - 10),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6))
                 
    ax1.set_title('Digital Twin Yield Projection under Heat Stress', fontsize=14)
    ax1.set_xlabel('Simulation Time (Ticks)', fontsize=12)
    ax1.set_ylabel('Estimated Yield (%)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, max(ideal_yield) + 5)
    
    plt.tight_layout()
    plt.savefig('fig_digital_twin.pdf')
    print("Generated fig_digital_twin.pdf")

if __name__ == "__main__":
    plot_dynamic_threshold()
    plot_decoupling()
    plot_digital_twin()
