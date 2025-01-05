#include <Servo.h>

Servo myServo;        // Create a servo object
const int servoPin = 9;  // Pin where the servo motor is connected
int currentAngle = 0;  // Current servo angle (default 0 degrees)

void setup() {
  Serial.begin(9600);      // Initialize serial communication
  myServo.attach(servoPin); // Attach the servo motor to pin 9
  myServo.write(currentAngle); // Initialize the servo to 0 degrees
  Serial.println("Arduino ready");
}

void loop() {
  // Check if data is available from the serial port
  if (Serial.available() > 0) {
    String receivedData = Serial.readStringUntil('\n'); // Read until newline
    receivedData.trim();  // Remove any whitespace or newline characters

    // Convert the received data to an integer (servo angle)
    int newAngle = receivedData.toInt();

    // Ensure the angle is valid (between 0 and 180 degrees)
    if (newAngle >= 0 && newAngle <= 180) {
      currentAngle = newAngle;
      myServo.write(currentAngle);  // Move the servo to the new angle
      Serial.print("Servo moved to: ");
      Serial.println(currentAngle);
    } else {
      Serial.println("Invalid angle received");
    }
  }
}
