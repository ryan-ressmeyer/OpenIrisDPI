
// These define's must be placed at the beginning before #include "TimerInterrupt.h"
// _TIMERINTERRUPT_LOGLEVEL_ from 0 to 4
// Don't define _TIMERINTERRUPT_LOGLEVEL_ > 0. Only for special ISR debugging only. Can hang the system.
#define TIMER_INTERRUPT_DEBUG         2
#define _TIMERINTERRUPT_LOGLEVEL_     0

#define USE_TIMER_1     true

// To be included only in main(), .ino with setup() to avoid `Multiple Definitions` Linker Error
#include "TimerInterrupt.h"

#if !defined(LED_BUILTIN)
  #define LED_BUILTIN     13
#endif

#define SYNC_PIN 3
#define NUM_RNG 8

#define TIMER_INTERVAL_MS    250
#define TIMER_FREQUENCY      (float) (1000.0f / TIMER_INTERVAL_MS)

void TimerHandler()
{
  // debug printing
#if (TIMER_INTERRUPT_DEBUG > 1)
  static int prev_t = 0;
  int t = millis();
  Serial.print("ITimer1 called, millis() = "); Serial.println(t - prev_t);
  prev_t = t;
#endif

  static bool sync_state = false;
  static bool next_random = false;
  static int irq_counter = 1; // each counter is .25s
  // .5 second interval
  // Then 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5

  // If the line is high from the previous interrupt set it low
  if (sync_state)
    sync_state = false;

  if (irq_counter <= 0)
  {
    sync_state = true;
    // Load the next interval
    if (next_random) {
      // Pick a number between 1 and 8
      irq_counter = 2 * (int) random(2, 2+NUM_RNG);
      next_random = false;
    } else {
      // Set counter
      irq_counter = 2;
      next_random = true;
    }
#if (TIMER_INTERRUPT_DEBUG > 0)
  Serial.print("Next Delay: "); Serial.println(irq_counter / TIMER_FREQUENCY);
#endif
  
  }

  
#if (TIMER_INTERRUPT_DEBUG > 2)
  Serial.println("TimerHandler State:");
  Serial.print("Line: "); Serial.print(sync_state); Serial.print(" | Counter: "); Serial.println(irq_counter);
#endif
  
  digitalWrite(LED_BUILTIN, sync_state);
  digitalWrite(SYNC_PIN, sync_state);
  irq_counter--;
  // 
}



void setup() {
  pinMode(LED_BUILTIN,  OUTPUT);
  pinMode(SYNC_PIN,  OUTPUT);
  randomSeed(analogRead(0));
  
  Serial.begin(115200);

  Serial.print(F("\nStarting Sync Line: "));
  Serial.println(BOARD_TYPE);
  Serial.println(TIMER_INTERRUPT_VERSION);
  Serial.print(F("CPU Frequency = ")); Serial.print(F_CPU / 1000000); Serial.println(F(" MHz"));

  // Timer0 is used for micros(), millis(), delay(), etc and can't be used
  // Select Timer 1-2 for UNO, 0-5 for MEGA
  // Timer 2 is 8-bit timer, only for higher frequency
  ITimer1.init();

  if (ITimer1.attachInterruptInterval(TIMER_INTERVAL_MS, TimerHandler, 0)) // Interval, Handle, Parameters, Duration (0 for infinite)
  {
    Serial.print(F("Starting  ITimer1 OK, millis() = ")); Serial.println(millis());
  }
  else
    Serial.println(F("Can't set ITimer1. Select another freq. or timer"));
}

void loop() {
  // put your main code here, to run repeatedly:

}
