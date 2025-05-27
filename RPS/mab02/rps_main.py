import os
import sys

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    clear()
    print("🎮 Welcome")
    print("---------")
    print("1. Play with Keyboard (Type r/p/s)")
    print("2. Play with Camera (Hand Gestures)")
    print("0. Exit")

    choice = input("\nSelect an option (0-2): ").strip()

    if choice == "1":
        from interface.play_typing import main as play_typing
        play_typing()
    elif choice == "2":
        from interface.play_camera import main as play_camera
        play_camera()
    elif choice == "0":
        print("👋 Goodbye!")
        sys.exit(0)
    else:
        print("Invalid option. Try again..")




if __name__ == "__main__":
    main()
