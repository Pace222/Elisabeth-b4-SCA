#define MAX_INPUT_SIZE 65536
#define START '<'
#define DELIMITER ','
#define STOP '>'

#define MAX_MESSAGE_SIZE 256

int new_data;
char input_buffer[MAX_INPUT_SIZE + 1];
size_t process_head;

int hex2int(char ch) {
  if (ch >= '0' && ch <= '9')
      return ch - '0';
  if (ch >= 'A' && ch <= 'F')
      return ch - 'A' + 10;
  if (ch >= 'a' && ch <= 'f')
      return ch - 'a' + 10;
  return -1;
}

void fill_random_uint4_t(uint4_t* array, size_t length) {
  for (int i = 0; i < length; i++) {
    array[i] = uint4_new((uint8_t) rand());
  }
}

String read_until(char delimiter) {
  char* start = input_buffer + process_head;
  char c = input_buffer[process_head++];
  while (c != '\0' && c != delimiter) {
    c = input_buffer[process_head++];
  }

  if (c == delimiter) {
    input_buffer[process_head - 1] = '\0';
    return String(start);
  } else {
    process_head -= 1;
    return String();
  }
}

int fill_array_from_user_hex_bytes(uint8_t* array, size_t length, char expected_delimiter) {
  char c1, c2;
  int n1, n2;

  // Skip first nibble if array expects an odd number of nibbles.
  if (length % 2 == 1) {
    c1 = input_buffer[process_head];
    if (c1 == '\0' || hex2int(c1) < 0) {
      return 0;
    } else {
      process_head += 1;
    }
  }

  for (int i = 0; i < length; i++) {
    c1 = input_buffer[process_head];
    c2 = input_buffer[process_head + 1];
    if (c1 == '\0' || c2 == '\0') {
      return i;
    } else {
      n1 = hex2int(c1);
      n2 = hex2int(c2);
      if (n1 < 0 || n2 < 0) {
        return i;
      }

      array[i] = (n1 << 4) + n2;

      process_head += 2;
    }
  }

  c1 = input_buffer[process_head];
  if (c1 == '\0') {
    return length;
  }
  if (c1 != expected_delimiter) {
    return -1;
  }
  process_head += 1;
  return length;
}

int fill_array_from_user_hex(uint4_t* array, size_t length, char expected_delimiter) {
  char c;
  int n;

  // Skip first nibble if array expects an odd number of nibbles.
  if (length % 2 == 1) {
    c = input_buffer[process_head];
    if (c == '\0' || hex2int(c) < 0) {
      return 0;
    } else {
      process_head += 1;
    }
  }

  for (int i = 0; i < length; i++) {
    c = input_buffer[process_head];
    if (c == '\0') {
      return i;
    } else {
      n = hex2int(c);
      if (n < 0) {
        return i;
      }

      array[i] = uint4_new(n);

      process_head += 1;
    }
  }

  c = input_buffer[process_head];
  if (c == '\0') {
    return length;
  }
  if (c != expected_delimiter) {
    return -1;
  }
  process_head += 1;
  return length;
}

int fill_array_from_user_until(uint4_t* array, char delimiter) {
  char c;
  int n;
  int length;

  for (length = 0; length < MAX_MESSAGE_SIZE; length++) {
    c = input_buffer[process_head];
    if (c == delimiter) {
      process_head += 1;
      return length;
    } else if (c == '\0') {
      return length;
    } else {
      n = hex2int(c);
      if (n < 0) {
        return -1;
      }

      array[length] = uint4_new(n);

      process_head += 1;
    }
  }

  c = input_buffer[process_head];
  if (c == delimiter) {
    process_head += 1;
    return length;
  } else if (c == '\0') {
    return length;
  }
  
  return -1;
}

void print_format(int mode, String choice, String arguments, String description) {
  String str_to_print = "Format: <";

  if (mode == 1) {
    str_to_print += "[4]" + String(DELIMITER);
  } else if (mode == 0) {
    str_to_print += "[B4]" + String(DELIMITER);
  } else {
    str_to_print += "[4 | B4]" + String(DELIMITER);
  }

  if (choice != "\0") {
    str_to_print += "[" + choice + "]" + String(DELIMITER);
  } else {
    str_to_print += "[0-9]" + String(DELIMITER);
  }

  str_to_print += arguments + "> | " + description;

  str_to_print += " All arguments must be in hexadecimal format.";

  Serial.println(str_to_print);

  if (choice == "\0") {
    Serial.println("  0: Benchmark whitening");
    Serial.println("  1: Benchmark single block of filter function");
    Serial.println("  2: Benchmark full filter function");
    Serial.println("  3: Benchmark whitening + full filter function");
    Serial.println("  4: Benchmark final addition with plaintext (encryption)");
    Serial.println("  5: Benchmark final subtraction with ciphertext (decryption)");
    Serial.println("  6: Benchmark complete encryption, single element");
    Serial.println("  7: Benchmark complete decryption, single element");
    Serial.println("  8: Benchmark complete encryption, full message");
    Serial.println("  9: Benchmark complete decryption, full message");
  }
}

void recv_input() {
  static int recv_in_progress = 0;
  static int truncated = 0;
  static size_t head = 0;
  char c;

  while (Serial.available() > 0 && !new_data) {
    c = Serial.read();

    if (recv_in_progress) {
      if (c != STOP) {
        input_buffer[head] = c;
        head++;
        if (head > MAX_INPUT_SIZE) {
          head = MAX_INPUT_SIZE;
          truncated = 1;
        }
      } else {
        input_buffer[head] = '\0'; // terminate the string
        recv_in_progress = 0;
        head = 0;
        if (truncated) {
          Serial.println("Input too long.");
          print_format(-1, "\0", "arg1,arg2,arg3,...", "Arguments depend on the benchmark.");
          truncated = 0;
        } else {
          new_data = 1;
        }
      }
    } else if (c == START) {
      recv_in_progress = 1;
    }
  }
}