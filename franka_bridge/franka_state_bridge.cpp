#include <iomanip>
#include <iostream>
#include <string>

#include <franka/exception.h>
#include <franka/robot.h>

// Persistent read-only bridge. Each input line requests one measured RobotState.
// O_T_EE is converted from libfranka column-major storage to a row-major JSON array.
int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <robot-hostname-or-ip>\n";
    return 2;
  }

  try {
    franka::Robot robot(argv[1], franka::RealtimeConfig::kIgnore);
    std::cout << "READY" << std::endl;

    std::string command;
    while (std::getline(std::cin, command)) {
      if (command == "q" || command == "Q") {
        break;
      }

      const franka::RobotState state = robot.readOnce();
      std::cout << std::setprecision(17) << "[";
      bool first = true;
      for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
          if (!first) {
            std::cout << ",";
          }
          first = false;
          std::cout << state.O_T_EE[col * 4 + row];
        }
      }
      std::cout << "]" << std::endl;
    }
  } catch (const franka::Exception& exception) {
    std::cerr << "Franka exception: " << exception.what() << "\n";
    return 1;
  }
  return 0;
}

