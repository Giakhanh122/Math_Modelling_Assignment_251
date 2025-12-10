from source import run

test_case1 = ("report.pnml", [1,2,3])

test_case2 = ("medium_deadlock.pnml", [1, 2, 3, 4, 5, 6, 7, 8])

test_case3 = ("medium_no_deadlock.pnml", [5, 1, 1, 1, 1, 1, 1, 2])

run(test_case1)
run(test_case2)
run(test_case3)
