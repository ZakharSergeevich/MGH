int echoPin = 2; 
int trigPin = 3; 
 
void setup() { 
    Serial.begin (115200); 
    pinMode(trigPin, OUTPUT); 
    pinMode(echoPin, INPUT); 
} 
 
void loop() { 
    int duration, cm; 
    digitalWrite(trigPin, LOW); 
    delayMicroseconds(2); 
    digitalWrite(trigPin, HIGH); 
    delayMicroseconds(10); 
    digitalWrite(trigPin, LOW); 
    duration = pulseIn(echoPin, HIGH); 
    cm = duration / 58;
    byte low = cm;
    byte high = cm >> 8;
    //Serial.write(high);
    //Serial.print("b");
    Serial.println(cm);
    //Serial.print("cm");
    //Serial.println(cm);
    //Serial.println(" cm"); 
    delay(100);
}
