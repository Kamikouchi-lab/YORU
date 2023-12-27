void setup() {
  Serial.begin(9600);
  pinMode(13, OUTPUT);
  digitalWrite(13, LOW);  //  初期化
}

void loop() {
  byte var;
  var = Serial.read();
  switch(var){
    case '0':
      digitalWrite(13, LOW);
      
      
      break;
    case '1':
      digitalWrite(13, HIGH);
      
      
      break;
    default:
      break;
  }
}
