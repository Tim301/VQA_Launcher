#include <TinyGPS++.h>
#include <Wire.h>
#include <LiquidTWI2.h>
LiquidTWI2 lcd(0x20);

const uint8_t GMT = +2;

char data;
uint8_t Second;
uint8_t last_second;
uint8_t Minute;
uint8_t Hour;
uint8_t Day;
uint8_t Month;
uint8_t Year;

TinyGPSPlus gps;

void setup() {
  DDRB = DDRB | B00110000;
  DDRC = DDRC | B11000000;
  DDRD = DDRD | B11011100;
  DDRF = DDRF | B11111111;
  
  lcd.setMCPType(LTI_TYPE_MCP23008); // must be called before begin()
  lcd.begin(16, 2);
  lcd.setBacklight(HIGH); // only supports HIGH or LOW
  Serial1.begin(9600);
  Serial.begin(9600);
}

void loop() {
  while (Serial1.available()) {
    data = Serial1.read();
    //  Serial.print(data);
    gps.encode(data);

    if (gps.time.isValid())
    {
      Second = gps.time.second();
      Minute = gps.time.minute();
      Hour   = gps.time.hour() + GMT;
    }

    // get date drom GPS module
    if (gps.date.isValid())
    {
      Day   = gps.date.day();
      Month = gps.date.month();
      Year  = gps.date.year();
    }

    if (last_second != gps.time.second()) // if time has changed
    {
      last_second = gps.time.second();
      print_time();

      if (check_schedule())
      {
        start_rec();
      } else {
        stop_rec();
      }
    }
  }
}

bool check_schedule() {
  bool must_rec = false;
  if ((Hour == 18 ) && (Minute >= 2 && Minute < 3)) {
    must_rec = true;
  }
  if ((Hour == 18 ) && (Minute >= 4 && Minute < 5)) {
    must_rec = true;
  }
  if ((Hour == 18 ) && (Minute >= 6 && Minute < 7)) {
    must_rec = true;
  }
  return must_rec;
}

void start_rec() {
  rec_all();
  lcd.setCursor(0, 1);
  lcd.print("Recording");
}

void stop_rec() {
  stop_all();
  lcd.setCursor(0, 1);
  lcd.print("                ");
}

void print_time() {
  String out;
  String str_hour = String(Hour);
  String str_minute = String(Minute);
  String str_second = String(Second);
  if (str_hour.length() == 1) {
    str_hour = "0" + str_hour;
  }
  if (str_minute.length() == 1) {
    str_minute = "0" + str_minute;
  }
  if (str_second.length() == 1) {
    str_second = "0" + str_second;
  }
  out = str_hour + ":" + str_minute + ":" + str_second;
  lcd.setCursor(0, 0);
  lcd.print(out);
}

void rec_all() {
   PORTB = B00110000;
   PORTC = B11000000;
   PORTD = B11011100;
   PORTF = B11111111;
}

void stop_all() {
  PORTC = B00000000;
  PORTD = B00000000;
  PORTD = B00000000;
  PORTF = B00000000;
}
