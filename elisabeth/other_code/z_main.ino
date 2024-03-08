void setup() {
  // Initialize serial and wait for port to open.
  Serial.begin(115200);
  while (!Serial)
    ;

  new_data = 0;

  init_sboxes_4();
  init_sboxes_b4();

  // Set the trigger PIN
  pinMode(TriggerPQ, OUTPUT);

  print_format(-1, "\0", "arg1,arg2,arg3,...", "Arguments depend on the benchmark.");
}


void loop() {
  recv_input();
  process_input();
}

void process_input() {
  if (new_data) {
    process_head = 0;
    new_data = 0;

    if (setup_mode()) {
      return;
    }
    
    if (setup_choice()) {
      return;
    }

    if (choice == "0") {
      // Benchmark whitening
      scenario_whitening_seed();
    } else if (choice == "1") {
      // Benchmark single block of filter function
      scenario_filter_block();
    } else if (choice == "2") {
      // Benchmark full filter function
      scenario_filter();
    } else if (choice == "3") {
      // Benchmark whitening + full filter function
      scenario_whitening_and_filter();
    } else if (choice == "4") {
      // Benchmark final addition with plaintext (encryption)
      scenario_addition();
    } else if (choice == "5") {
      // Benchmark final subtraction with ciphertext (decryption)
      scenario_subtraction();
    } else if (choice == "6") {
      // Benchmark complete encryption, single element
      scenario_encrypt_elem_seed();
    } else if (choice == "7") {
      // Benchmark complete decryption, single element
      scenario_decrypt_elem_seed();
    } else if (choice == "8") {
      // Benchmark complete encryption, full message
      scenario_encrypt_message_seed();
    } else if (choice == "9") {
      // Benchmark complete decryption, full message
      scenario_decrypt_message_seed();
    } else {
      print_format(mode, "\0", "arg1,arg2,arg3,...", "Arguments depend on the benchmark.");
    }
  }
}
