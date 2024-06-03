/*
 * TODO:
 * 
 * 2. examine the output range of tremolo and chorus. Are they [0,255]?
 * 3. redundant sine tables for ring_mod.
 * 4. Serial buffer length = 63 Bytes available. try to find ways to lengthen it.
 */


#include <avr/pgmspace.h>
#include <math.h>

const uint8_t buffer_length = 60; //serial buffer length = 64bytes. 63 Bytes available for TX.
uint8_t audio_buffer[buffer_length + 2]; //initialise buffer. 2nd last byte:current mode. Last byte: current buffer index
uint8_t buffer_index = 0; //change this to uint16_t when length>256, and change the upload code.

unsigned long last_serial_time = 0;
unsigned long serial_upload_interval = 30*1000; //upload every 30ms

const int button_up = 3;
const int button_down = 2;
volatile uint8_t effect_code = 0; //the current code of effect. range = 0 to effect_numbers-1
volatile uint8_t effect_numbers = 6; //change this for the total number of effects we have.

void DACoutput(int value){    //value=[0,255], pin 6 is LSB, pin 13 is MSB
  int PB = value >> 2;
  PORTB = PB;
  
  int VD = PORTD;
  int PD = value & B00000011;
  PD = PD << 6;
  VD = VD & B00111111;
  VD = VD | PD;
  PORTD = VD;  
}

//for all effects:
const int sampleRate = 33000; // Sample rate in Hz
const int lfoResolution = 256; // Resolution of the LFO sine wave
unsigned long currentTime=0, lastTime=0, deltaTime=0; //put them as global variables
float phase = 0.0; // Phase of the LFO

//for Tremolo:
const float tremoloFrequency = 5.0; // Tremolo frequency in Hz
const float tremoloDepth = 0.5; // Tremolo depth (0.0 to 1.0)

//for bitcrusher:
const int bitDepth = 4; // Bit depth for bitcrusher effect

// Precomputed sine wave
const uint8_t sineTable[lfoResolution] PROGMEM = {
  127, 130, 133, 136, 139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172,
  175, 178, 181, 184, 187, 190, 193, 196, 199, 202, 205, 208, 211, 214, 217, 220,
  223, 226, 229, 231, 234, 237, 240, 243, 246, 249, 251, 254, 257, 260, 263, 265,
  268, 271, 273, 276, 279, 281, 284, 287, 289, 292, 294, 297, 299, 302, 304, 307,
  309, 312, 314, 316, 319, 321, 323, 326, 328, 330, 332, 335, 337, 339, 341, 343,
  345, 347, 349, 351, 353, 355, 357, 359, 361, 363, 364, 366, 368, 370, 371, 373,
  375, 376, 378, 380, 381, 383, 384, 386, 387, 389, 390, 391, 393, 394, 395, 396,
  398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413,
  413, 414, 415, 416, 417, 417, 418, 419, 419, 420, 421, 421, 422, 422, 423, 423,
  424, 424, 424, 425, 425, 425, 426, 426, 426, 426, 426, 426, 426, 426, 426, 426,
  426, 426, 426, 426, 426, 426, 426, 426, 426, 426, 426, 426, 426, 426, 426, 426,
  426, 426, 426, 426, 426, 426, 426, 426, 426, 426, 426, 426, 426, 426, 426, 426,
  426, 426, 426, 426, 426, 426, 426, 426, 426, 426, 426, 426, 426, 426, 426, 426
};

void setup() {
  Serial.begin(115200);
  
  //pinmode: 6-13 as output:
  DDRB = DDRB | B00111111;
  pinMode(6, OUTPUT);
  pinMode(7, OUTPUT);

  pinMode(button_up, INPUT);
  pinMode(button_down, INPUT);
  attachInterrupt(digitalPinToInterrupt(button_up), effect_next, RISING);
  attachInterrupt(digitalPinToInterrupt(button_down), effect_last, RISING);

  //configure ADC clock prescaler as 16. max ADC sampling rate 66kHz.
  ADCSRA &= ~(bit (ADPS0) | bit (ADPS1) | bit (ADPS2)); // clear prescaler bits
  ADCSRA |= bit (ADPS2);    //  16-scale 
}

void effect_next(){
  effect_code++;
  effect_code %= effect_numbers;
}

void effect_last(){
  effect_code--;
  effect_code %= effect_numbers;
}

void buffer_update(){
  buffer_index++;
  if(buffer_index >= buffer_length) buffer_index = 0; //this is faster than % operator
  audio_buffer[buffer_index] = analogRead(A0)>>2; //0-1023 -> 0-255
}

void bypass(){
  DACoutput(audio_buffer[buffer_index]);
}

void tremolo();
void chorus();
void distortion();
void bitcrusher();
void ring_mod();

void loop() {
  buffer_update();
  
  currentTime = micros();
  deltaTime = (currentTime - lastTime) / 1000000.0;
  lastTime = currentTime;

  if(last_serial_time - currentTime >= serial_upload_interval){
    audio_buffer[buffer_length] = effect_code;  //2nd last byte in the buffer is the effect code.
    audio_buffer[buffer_length+1]= buffer_index;  //2nd last byte in the buffer is the effect code.
    if(Serial.availableForWrite() >= buffer_length+1){
      Serial.write(audio_buffer, buffer_length+1);
    }
    //time taken by serial write could cause clicks in sound. Try to see if we need to re-update the time here.
  }
  
  switch(effect_code){
    case 0:
      bypass();
      break;
    
    case 1:
      tremolo();
      break;
    
    case 2:
      chorus();
      break;
    
    case 3:
      distortion();
      break;

    case 4:
      bitcrusher();
      break;

    case 5:
      ring_mod();
      break;
  }
}

void tremolo(){
  // Update LFO phase
  phase += tremoloFrequency * deltaTime;
  if (phase >= 1.0) phase -= 1.0;

  int index = (int)(phase * lfoResolution);
  float lfo = pgm_read_byte_near(sineTable + index) / 255.0; // LFO ranges from 0.0 to 1.0

  // Apply tremolo effect
  int tremoloValue = int(audio_buffer[buffer_index] * (1.0 - tremoloDepth + (lfo * tremoloDepth)));
  // Output the tremolo value to the DAC
  DACoutput(tremoloValue);
}

void chorus(){
  // Update LFO phase
  phase += 0.5 * deltaTime; // Chorus modulation frequency
  if (phase >= 1.0) phase -= 1.0;

  // Calculate LFO value (sine wave)
  int index = (int)(phase * lfoResolution);
  float lfo = pgm_read_byte_near(sineTable + index) / 255.0; // LFO ranges from 0.0 to 1.0

  // Apply chorus effect
  float mod_depth = 0.5; // Depth of modulation
  int mod_index = buffer_index - (int)(mod_depth * lfo * buffer_length);
  if (mod_index < 0) mod_index += buffer_length;

  int chorusValue = (audio_buffer[buffer_index] + audio_buffer[mod_index]) / 2;

  if (chorusValue > 255) {
    chorusValue = 255; // Clipping
  } else if (chorusValue < 0) {
    chorusValue = 0; // Clipping
  }
  DACoutput(chorusValue);
}

void distortion(){
  // Apply distortion effect
  //audio_buffer: uint8_t [0,255]. distortedValue: int [-32768,32767]. max gain = 128. Not a worry.
  float gain = 2; // Increase the gain for more distortion
  int distortedValue = (int)audio_buffer[buffer_index] * gain;
  if (distortedValue > 255) {
    distortedValue = 255; // Clipping
  } else if (distortedValue < 0) {
    distortedValue = 0; // Clipping
  }
  DACoutput(distortedValue);
}

void bitcrusher(){
  // Apply bitcrusher effect
  int stepSize = pow(2, 10 - bitDepth);
  int crushedValue = (audio_buffer[buffer_index] / stepSize) * stepSize;
  DACoutput(crushedValue);
}

void ring_mod(){
  
}
