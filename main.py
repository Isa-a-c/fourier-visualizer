"""소스 실행과 Windows 배포판의 공통 진입점입니다."""
import multiprocessing


def main():
    import sys
    from PySide6.QtWidgets import QApplication
    from engineering_math.ui.window import MainWindow
    application = QApplication(sys.argv)
    application.setApplicationName('EngineeringMathStudio')
    window = MainWindow()
    if '--self-test' in sys.argv:
        from engineering_math.selftest import run
        return run(application, window, sys.argv[sys.argv.index('--self-test')+1])
    window.show()
    from PySide6.QtCore import QTimer
    QTimer.singleShot(0, window.calculate)
    return application.exec()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    raise SystemExit(main())
