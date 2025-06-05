from backend.dataprocess import AcademicWarningGUI


def main():
    print("=== 学业预警系统 ===")
    gui = AcademicWarningGUI()
    gui.run()


if __name__ == "__main__":
    main()