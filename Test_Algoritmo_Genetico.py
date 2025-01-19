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

