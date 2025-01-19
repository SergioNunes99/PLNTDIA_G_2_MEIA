from functools import reduce

import Algoritmo_Genetico as ag


TEST_EXAMS_TYPES = ['BT','ECG']
TEST_PATIENT_PREFERENCE = {
1: [660, 0, 540, 540, 0, 630, 540],
2: [660, 660, 0, 1020, 0, 900, 0],
}

TEST_PATIENT_PRIORITY = {
1: 0.8,
2: 0.7,
}

TEST_MAX_PRIORITY = max(TEST_PATIENT_PRIORITY.values())
TEST_MIN_PRIORITY = min(TEST_PATIENT_PRIORITY.values())

TEST_PRIORITY_MAX= reduce(lambda a, b: a + b, [TEST_MAX_PRIORITY * x for x in range(0, len(TEST_PATIENT_PREFERENCE)* len(TEST_EXAMS_TYPES))])
TEST_PRIORITY_MIN= reduce(lambda a, b: a + b, [TEST_MIN_PRIORITY * x for x in range(0, len(TEST_PATIENT_PREFERENCE)* len(TEST_EXAMS_TYPES))])

def crossover_individuals():
    gene1 = ag.Gene(medical_personnel_id=1,
                                    appointment_minute_of_day_start=660,
                                    appointment_minute_of_day_end=670,
                                    exam_type="BT",
                                    weekday=0,
                                    patient_id=1,
                                    room=0)
    gene2 = ag.Gene(medical_personnel_id=1,
                 appointment_minute_of_day_start=670,
                 appointment_minute_of_day_end=680,
                 exam_type='ECG',
                 weekday=0,
                 patient_id=1,
                 room=1)
    gene3 = ag.Gene(medical_personnel_id=1,
                    appointment_minute_of_day_start=690,
                    appointment_minute_of_day_end=700,
                    exam_type='BT',
                    weekday=0,
                    patient_id=2,
                    room=0)
    gene4 = ag.Gene(medical_personnel_id=1,
                    appointment_minute_of_day_start=710,
                    appointment_minute_of_day_end=720,
                    exam_type='ECG',
                    weekday=0,
                    patient_id=2,
                    room=1)
    individual1_genes = [gene1, gene2, gene3, gene4]

    gene1 = ag.Gene(medical_personnel_id=2,
                                    appointment_minute_of_day_start=670,
                                    appointment_minute_of_day_end=870,
                                    exam_type="ECG",
                                    weekday=3,
                                    patient_id=0,
                                    room=4)
    gene2 = ag.Gene(medical_personnel_id=1,
                 appointment_minute_of_day_start=300,
                 appointment_minute_of_day_end=590,
                 exam_type='ECG',
                 weekday=0,
                 patient_id=0,
                 room=1)
    gene3 = ag.Gene(medical_personnel_id=0,
                    appointment_minute_of_day_start=500,
                    appointment_minute_of_day_end=670,
                    exam_type='ECG',
                    weekday=0,
                    patient_id=1,
                    room=3)
    gene4 = ag.Gene(medical_personnel_id=1,
                    appointment_minute_of_day_start=400,
                    appointment_minute_of_day_end=500,
                    exam_type='BT',
                    weekday=5,
                    patient_id=1,
                    room=5)
    individual2_genes = [gene1, gene2, gene3, gene4]

    return ag.Individual(genes = individual1_genes), ag.Individual(genes = individual2_genes)


def test_ensure_calculate_fitness_function_returns_expected_value():
    ag.EXAMS_TYPES=TEST_EXAMS_TYPES
    ag.EXAM_COUNT = len(TEST_EXAMS_TYPES)
    ag.PATIENT_COUNT = len(TEST_PATIENT_PREFERENCE)
    ag.PATIENT_PREFERENCE = TEST_PATIENT_PREFERENCE
    ag.PATIENT_PRIORITY = TEST_PATIENT_PRIORITY

    assert ag.EXAMS_TYPES == TEST_EXAMS_TYPES
    assert ag.EXAM_COUNT == len(TEST_EXAMS_TYPES)

    assert ag.PATIENT_COUNT == len(TEST_PATIENT_PREFERENCE)
    assert ag.PATIENT_PREFERENCE == TEST_PATIENT_PREFERENCE
    assert ag.PATIENT_PRIORITY == TEST_PATIENT_PRIORITY

    individual_genes = []
    gene1 = ag.Gene(medical_personnel_id=1,
                                    appointment_minute_of_day_start=660,
                                    appointment_minute_of_day_end=670,
                                    exam_type="BT",
                                    weekday=0,
                                    patient_id=1,
                                    room=0)
    individual_genes.append(gene1)
    gene2 = ag.Gene(medical_personnel_id=1,
                 appointment_minute_of_day_start=670,
                 appointment_minute_of_day_end=680,
                 exam_type='ECG',
                 weekday=0,
                 patient_id=1,
                 room=1)
    individual_genes.append(gene2)
    gene3 = ag.Gene(medical_personnel_id=1,
                    appointment_minute_of_day_start=690,
                    appointment_minute_of_day_end=700,
                    exam_type='BT',
                    weekday=0,
                    patient_id=2,
                    room=0)
    individual_genes.append(gene3)

    gene4 = ag.Gene(medical_personnel_id=1,
                    appointment_minute_of_day_start=710,
                    appointment_minute_of_day_end=720,
                    exam_type='ECG',
                    weekday=0,
                    patient_id=2,
                    room=1)
    individual_genes.append(gene4)

    test_individual = ag.Individual(genes= individual_genes)

    patient_exams_errors_penalization= ag.calculate_patient_exams_errors(test_individual)
    unscheduled_exams_due_lack_availability_penalization= ag.calculate_unscheduled_exams_due_lack_disponibility(test_individual)

    priority_assertiveness_penalization = ag.calculate_priority_assertiveness(test_individual, TEST_PRIORITY_MIN, TEST_PRIORITY_MAX)
    patient_preference_assertiveness_penalization=ag.calculate_patient_preference_assertiveness(test_individual)

    assert unscheduled_exams_due_lack_availability_penalization == 0
    assert priority_assertiveness_penalization == 0
    assert patient_exams_errors_penalization == 0
    assert patient_preference_assertiveness_penalization == 0

def test_crossover():
    ag.EXAMS_TYPES = TEST_EXAMS_TYPES
    ag.EXAM_COUNT = len(TEST_EXAMS_TYPES)
    ag.PATIENT_COUNT = len(TEST_PATIENT_PREFERENCE)
    ag.PATIENT_PREFERENCE = TEST_PATIENT_PREFERENCE
    ag.PATIENT_PRIORITY = TEST_PATIENT_PRIORITY
    ag.CROSSING_RATE = 1
    ag.CROSSOVER_MASK_GENE = [0,0,0,0,0,0,1]

    individual1_genes, individual2_genes = crossover_individuals()
    descendents = ag.crossover([individual1_genes, individual2_genes])

    assert descendents == 0


def mapPopulationToScheduelingInfo(population):
    [3, 495, 505, RX, 3, 9, room3, 8, 695, 705, BT, 0, 4, room1, 8, 700, 710, ECG, 6, 13, room2, 4, 895, 905, RX, 4, 9,
     room2, 2, 420, 450, ECO, 0, 11, room5, 6, 815, 835, AC, 3, 15, room1, 9, 600, 660, EG, 5, 1, room7, 3, 550, 560,
     BT, 4, 9, room1, 8, 900, 910, ECG, 6, 9, room2, 4, 780, 790, BT, 3, 2, room1, 8, 920, 930, BT, 3, 1, room4, 3, 785,
     805, AC, 2, 13, room1, 5, 835, 845, ECG, 4, 11, room2, 3, 900, 960, EG, 6, 11, room3, 5, 580, 610, ECO, 6, 12,
     room5, 7, 640, 670, ECO, 0, 9, room4, 10, 1050, 1080, ECO, 1, 8, room4, 7, 845, 905, EG, 1, 6, room3, 6, 980, 990,
     BT, 1, 3, room4, 5, 910, 940, ECO, 0, 4, room4, 7, 860, 880, AC, 4, 15, room1, 2, 940, 960, AC, 6, 3, room1, 9,
     785, 795, RX, 5, 15, room3, 7, 800, 810, BT, 2, 10, room4, 10, 845, 855, ECG, 6, 1, room2, 6, 700, 710, RX, 0, 13,
     room3, 8, 795, 855, EG, 2, 8, room3, 7, 820, 880, EG, 3, 4, room7, 3, 515, 525, BT, 4, 8, room1, 3, 435, 495, EG,
     3, 3, room3, 9, 635, 665, ECO, 5, 14, room4, 3, 810, 820, RX, 2, 2, room2, 5, 800, 860, EG, 0, 12, room3, 2, 690,
     700, BT, 0, 11, room4, 10, 640, 700, EG, 3, 10, room7, 7, 710, 720, RX, 0, 1, room3, 2, 555, 615, EG, 0, 1, room7,
     6, 605, 665, EG, 3, 2, room3, 2, 515, 535, AC, 3, 1, room1, 5, 570, 580, RX, 5, 3, room3, 2, 535, 595, EG, 3, 11,
     room3, 10, 820, 840, AC, 4, 14, room1, 7, 675, 685, RX, 6, 13, room2, 2, 640, 660, AC, 6, 11, room1, 8, 665, 685,
     AC, 6, 6, room1, 5, 790, 800, ECG, 6, 2, room6, 6, 640, 650, BT, 1, 13, room4, 6, 1065, 1075, RX, 1, 7, room3, 5,
     795, 805, RX, 3, 11, room2, 8, 805, 815, ECG, 6, 3, room6, 3, 885, 895, BT, 2, 11, room4, 10, 815, 825, RX, 5, 2,
     room2, 4, 785, 815, ECO, 2, 9, room5, 8, 875, 885, RX, 2, 8, room3, 10, 705, 715, BT, 6, 14, room4, 2, 780, 790,
     ECG, 6, 10, room2, 4, 800, 810, RX, 4, 1, room2, 8, 665, 695, ECO, 1, 10, room5, 4, 930, 940, BT, 2, 4, room4, 4,
     690, 700, RX, 6, 4, room2, 8, 705, 715, RX, 3, 5, room2, 6, 785, 795, RX, 4, 6, room2, 5, 710, 720, ECG, 4, 7,
     room2, 10, 835, 895, EG, 0, 14, room7, 6, 635, 655, AC, 2, 5, room1, 6, 625, 635, BT, 0, 9, room4, 10, 655, 685,
     ECO, 4, 2, room4, 4, 620, 650, ECO, 5, 15, room4, 10, 955, 1015, EG, 4, 1, room3, 7, 965, 975, ECG, 4, 13, room2,
     7, 1045, 1055, ECG, 1, 12, room6, 7, 600, 660, EG, 6, 13, room7, 6, 600, 620, AC, 6, 8, room1, 6, 905, 935, ECO, 1,
     7, room4, 2, 590, 610, AC, 4, 2, room1, 4, 675, 685, RX, 5, 9, room3, 9, 945, 975, ECO, 0, 5, room4, 6, 1055, 1075,
     AC, 3, 12, room1, 6, 660, 680, AC, 1, 1, room1, 3, 785, 795, ECG, 0, 5, room2, 3, 615, 625, RX, 4, 11, room3, 5,
     690, 700, RX, 5, 1, room3, 5, 815, 825, ECG, 6, 14, room6, 2, 480, 490, ECG, 1, 8, room6, 4, 710, 720, RX, 6, 12,
     room2, 4, 885, 915, ECO, 0, 14, room5, 9, 955, 985, ECO, 2, 15, room4, 4, 790, 800, RX, 6, 4, room2, 6, 810, 820,
     RX, 1, 7, room3, 7, 815, 845, ECO, 6, 13, room5]