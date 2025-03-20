
from example_single import exercise_single
from example_multiple import exercise_multiple
from exercise0 import exercise0
from exercise1 import exercise1
from exercise2 import exercise2
from exercise3 import exercise3
from exercise4 import exercise4
import plot_results
import farms_pylog as pylog


def main():

    pylog.info("Running Project 1 exercises")

    exercise_single()
    exercise_multiple()
    exercise0()
    exercise1()
    exercise2()
    exercise3()
    exercise4()
    plot_results.main()


if __name__ == '__main__':
    main()

