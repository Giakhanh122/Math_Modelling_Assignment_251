from source import run

test_case1 = ("report.pnml", [1,2,3], True)

test_case2 = ("medium_deadlock.pnml", [1, 2, 3, 4, 5, 6, 7, 8], True)

test_case3 = ("medium_no_deadlock.pnml", [5, 1, 1, 1, 1, 1, 1, 2], True)


run(test_case1)

