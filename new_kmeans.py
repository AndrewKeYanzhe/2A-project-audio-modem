    def process_signal(self):
        n_bins = 4096
        self.received_signal = self.load_data(self.received_file)
        # Remove cyclic prefix and get blocks
        blocks = self.remove_cyclic_prefix(self.received_signal)
        # Define subcarrier frequencies for OFDM
        subcarrier_frequencies = np.fft.fftfreq(self.block_size, d=1/self.fs)
        # Interpolate the frequency response to match the subcarrier frequencies
        interpolated_response = self.interpolate_frequency_response(subcarrier_frequencies)
        complete_binary_data = ''
        # Get the frequency bins corresponding to the given frequency range
        bin_low,bin_high = cut_freq_bins(self.f_low, self.f_high, self.fs, self.block_size) 

        print("number of blocks:",len(blocks))
        for index, block in enumerate(blocks):
            if index == 0 and use_pilot_tone:#operating the first block, get gn
                print("using pilot tone")
                np.random.seed(1)
                constellation_points = np.array([1+1j, -1+1j, -1-1j, 1-1j])
                symbols_extended = np.random.choice(constellation_points, n_bins)

                symbols_extended[0] = 0
                symbols_extended[n_bins // 2] = 0
                symbols_extended[n_bins//2+1:] = np.conj(np.flip(symbols_extended[1:n_bins//2]))
                pilot_n = symbols_extended
                r_n = self.apply_fft(block, self.block_size)
                pilot_response = r_n/pilot_n
                self.g_n = pilot_response
                frequencies=subcarrier_frequencies
                phase_response = np.angle(self.g_n, deg=True)
                # Plot the phase response
                plt.figure(figsize=(6, 4))
                plt.plot(frequencies, phase_response)
                plt.title('Phase Response of the Channel (Pilot symbol)')
                plt.xlabel('Frequency (Hz)')
                plt.xlim(0, 10000)
                plt.ylabel('Phase (Degrees)')
                plt.show()
            # Apply FFT to the block
            r_n = self.apply_fft(block, self.block_size)
            if use_pilot_tone == False:
                self.g_n = interpolated_response
            # Compensate for the channel effects
            x_n = self.channel_compensation(r_n, self.g_n)
            # Save the constellation points for plotting
            self.received_constellations.extend(r_n[bin_low:bin_high+1])
            self.compensated_constellations.extend(x_n[bin_low:bin_high+1]) 

        self.plot_constellation(self.received_constellations, title="Constellation\nBefore Compensation")
        self.plot_constellation(self.compensated_constellations, title="Constellation\nAfter Compensation")
        # self.plot_constellation(compensated_constellations_subsampled, title="Constellation After Compensation,\nsubsampled 1:10")


        for index, block in enumerate(blocks):
            # Apply FFT to the block
            
            r_n = self.apply_fft(block, self.block_size)
            if use_pilot_tone == False:
                self.g_n = interpolated_response
            # Compensate for the channel effects
            x_n = self.channel_compensation(r_n, self.g_n)

            constellations = np.copy(x_n[bin_low:bin_high+1])
            shifted_constellations = self.apply_kmeans(constellations, n_clusters=4, random_state=42)
  
            
            if shift_constellation_phase:
                binary_data = self.qpsk_demapper(shifted_constellations) 
                binary_data = self.qpsk_demapper(constellations) 


            

            if use_ldpc:
                # if index == 0 and use_pilot_tone:
                #     continue


                block_length = len(binary_data)
                ldpc_encoded_length = (block_length//24)*24

                ldpc_signal = binary_data[0:ldpc_encoded_length]

                # print(list(ldpc_signal))

                #convert string to list
                ldpc_signal_list = np.array([int(element) for element in list(ldpc_signal)])

                # print(ldpc_signal_list)

                ldpc_decoded, ldpc_decoded_with_redundancies = decode_ldpc(ldpc_signal_list)

                
                #convert list to string
                ldpc_decoded = ''.join(str(x) for x in ldpc_decoded)

                complete_binary_data += ldpc_decoded

            elif use_ldpc == False:
                if index != 0:
                    complete_binary_data += binary_data

        logging.info(f"Recovered Binary Data Length: {len(complete_binary_data)}")
        return complete_binary_data
    
    def apply_kmeans(self, compensated_constellations, n_clusters=5, random_state=42):

        # Convert complex numbers to a 2D array of their real and imaginary parts
        data = np.array([[z.real, z.imag] for z in compensated_constellations])
        
        # Apply k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=random_state).fit(data)
        
        # Get the cluster centroids
        centroids = kmeans.cluster_centers_
        
        # Sort the top 4 centroids by magnitude
        top_4 = sorted(centroids, key=lambda c: c[0]**2 + c[1]**2, reverse=True)[:4]
        
        # Calculate phases and sort them
        phases = [(c, math.atan2(c[1], c[0])) for c in top_4]
        phases_sorted = sorted(phases, key=lambda x: x[1])
        
        # Calculate the sum of angles in degrees
        sum_angles = 0
        for c, angle in phases_sorted:
            angle_degrees = math.degrees(angle)
            if angle_degrees < 0:
                angle_degrees += 360
            sum_angles += angle_degrees
        
        # Calculate the phase shift needed
        phase_shift_needed = (720 - sum_angles) / 4
        
        # Convert centroids back to complex numbers
        centroid_complex_numbers = [complex(c[0], c[1]) for c in centroids]
        
        # Apply the phase shift to the original constellations
        shifted_constellations = [z * cmath.exp(1j * math.radians(phase_shift_needed)) for z in compensated_constellations]
        
        return shifted_constellations