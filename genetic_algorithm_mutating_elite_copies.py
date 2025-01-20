# Paper do algoritmo genetico estudado
#
# Problema a ser resolvido: numero minimmo de recursos humanos usados num hospital para atender o maximo de pacientes possiveis
#
# Funçao fitness: MinZ = (x11 + x12 + x13 + x21 + x22 + ... + x91 + x92 + x93)
# Casa variavel de decisao (x11, x12, ...) tem dois atributos, que representam a especialidade do medico e o turno, respetivamente
# Cada especialista tem um tempo diferente de atuação em cada turno, é isto que queremos otimizar
#
# Existem varias restriçoes que têm de ser tidas em conta aquando da geraçao de novos individuos
#
# Aqui, a genetica é composta por 27 variaveis, cada variavel composta por 2 atributos
import copy
import csv
import datetime
import math
import random
import string
from copy import deepcopy
from functools import reduce

import pandas as pd
from matplotlib import pyplot as plt, table

# ============================================================================================================
# Algorithm optimization parameters

CROSSING_RATE = 0.7
SUBSTITUTION_RATE = 0.7
MUTATION_RATE = 0.8
PITY_SELECTION_RATE = 0.10


# ============================================================================================================

POPULATION_SIZE = 100
NUMBER_OF_GENERATIONS = 2000

# EXAMS
EXAMS_TYPES = ['BT', 'ECG', 'EG', 'AC', 'ECO', 'RX']

EXAMS_DURATION = {'BT': 10,
                  'ECG': 10,
                  'EG': 60,
                  'AC': 20,
                  'ECO': 30,
                  'RX': 10}

EXAM_COUNT = len(EXAMS_TYPES)

# ================================ FITNESS =======================================================================

FITNESS_ERRORS_WEIGHT = 4
FITNESS_OVERLAPS_WEIGHT = 4
FITNESS_UNSCHEDULED_EXAMS_WEIGHT = 0.3
FITNESS_PRIORITY_WEIGHT = 0.3
FITNESS_PREFERENCE_WEIGHT = 0.1

# ================================================================================================================
# METRICS
average_fitness_values = []
best_fitness_values = []

# ==============================================================================================================
# ROOMS
ROOMS = ['room1', 'room2', 'room3', 'room4', 'room5', 'room6', 'room7']

ROOMS_AVAILABLE = {'BT': [ROOMS[0], ROOMS[3]],
                   'ECG': [ROOMS[1], ROOMS[5]],
                   'EG': [ROOMS[2], ROOMS[6]],
                   'AC': [ROOMS[0]],
                   'ECO': [ROOMS[4], ROOMS[3]],
                   'RX': [ROOMS[1], ROOMS[2]]}

ROOM_FULL_DISPONIBILITY = [[0, 1440]]

# PATIENTS
# Patients represented by sequential, ordered numbers, initiating in 1!!
PATIENT_PREFERENCE = {
    1: [660, 0, 540, 540, 0, 630, 540],
    2: [660, 660, 0, 1020, 0, 900, 0],
    3: [0, 0, 630, 0, 960, 0, 540],
    4: [840, 0, 0, 0, 960, 630, 480],
    5: [0, 630, 0, 960, 840, 0, 0],
    6: [0, 0, 540, 0, 0, 540, 0],
    7: [780, 480, 0, 0, 0, 0, 480],
    8: [0, 0, 1020, 0, 900, 0, 630],
    9: [660, 0, 0, 840, 0, 540, 0],
    10: [0, 840, 0, 0, 900, 540, 0],
    11: [480, 0, 960, 0, 540, 0, 480],
    12: [960, 0, 0, 960, 540, 540, 0],
    13: [1020, 480, 0, 0, 0, 0, 0],
    14: [0, 0, 540, 0, 540, 540, 540],
    15: [630, 0, 0, 960, 0, 540, 540],
    #    16: [0, 720, 900, 810, 0, 930, 870],
    #    17: [710, 750, 630, 0, 720, 850, 900],
    #    18: [850, 890, 780, 0, 910, 0, 810],
    #    19: [630, 0, 810, 730, 780, 920, 840],
    #    20: [900, 930, 0, 880, 770, 0, 810],
    #    21: [830, 0, 0, 920, 870, 890, 660],
    #    22: [630, 0, 960, 700, 910, 810, 700],
    #    23: [730, 810, 860, 720, 840, 760, 930],
    #    24: [900, 960, 0, 850, 0, 0, 660],
    #    25: [770, 920, 890, 0, 700, 930, 900],
    #    26: [960, 0, 0, 0, 910, 780, 840],
    #    27: [810, 750, 700, 730, 760, 920, 870],
    #    28: [870, 690, 0, 840, 910, 930, 660],
    #    29: [930, 810, 0, 890, 770, 880, 660],
    #    30: [880, 700, 630, 0, 810, 960, 900],
    #    31: [660, 870, 750, 900, 910, 850, 770],
    #    32: [960, 0, 780, 880, 730, 920, 810],
    #    33: [920, 0, 0, 0, 0, 930, 810],
    #    34: [780, 0, 810, 0, 720, 0, 900],
    #    35: [810, 850, 890, 760, 720, 930, 700],
    #    36: [630, 0, 0, 0, 870, 850, 730],
    #    37: [0, 810, 700, 660, 0, 760, 960],
    #    38: [700, 0, 930, 750, 700, 910, 840],
    #    39: [760, 960, 850, 660, 730, 810, 880],
    #    40: [850, 780, 700, 0, 870, 920, 930],
    #    41: [900, 890, 660, 0, 0, 870, 960],
    #    42: [910, 830, 770, 660, 700, 840, 720],
    #    43: [930, 810, 850, 0, 720, 910, 780],
    #    44: [960, 770, 720, 660, 870, 880, 930],
    #    45: [810, 690, 880, 810, 920, 750, 960],
    #    46: [0, 730, 0, 760, 0, 810, 900],
    #    47: [840, 720, 890, 900, 850, 780, 930],
    #    48: [900, 880, 0, 700, 770, 720, 660],
    #    49: [930, 0, 780, 920, 850, 840, 960],
    #    50: [0, 720, 900, 810, 630, 930, 870],
}

PATIENT_PRIORITY = {
    1: 0.7,
    2: 0.8,
    3: 0.5,
    4: 0.7,
    5: 0.8,
    6: 0.6,
    7: 0.6,
    8: 0.9,
    9: 0.8,
    10: 0.9,
    11: 0.6,
    12: 0.7,
    13: 0.5,
    14: 0.8,
    15: 0.5,
    #    16: 0.7,
    #    17: 0.8,
    #    18: 0.9,
    #    19: 0.6,
    #    20: 0.9,
    #    21: 0.7,
    #    22: 0.5,
    #    23: 0.6,
    #    24: 0.9,
    #    25: 0.7,
    #    26: 0.8,
    #    27: 0.6,
    #    28: 0.9,
    #    29: 0.5,
    #    30: 0.8,
    #    31: 0.7,
    #    32: 0.8,
    #    33: 0.6,
    #    34: 0.9,
    #    35: 0.5,
    #    36: 0.8,
    #    37: 0.7,
    #    38: 0.6,
    #    39: 0.5,
    #    40: 0.8,
    #    41: 0.9,
    #    42: 0.7,
    #    43: 0.8,
    #    44: 0.6,
    #    45: 0.7,
    #    46: 0.8,
    #    47: 0.6,
    #    48: 0.9,
    #    49: 0.5,
    #    50: 0.8,
}

PATIENT_COUNT = len(PATIENT_PREFERENCE)
INDIVIDUAL_GENES_COUNT = PATIENT_COUNT * EXAM_COUNT

MAX_PRIORITY = max(PATIENT_PRIORITY.values())
MIN_PRIORITY = min(PATIENT_PRIORITY.values())

PRIORITY_MAX = (1 - MIN_PRIORITY) * INDIVIDUAL_GENES_COUNT
PRIORITY_MIN = 0

priorities_array = list(PATIENT_PRIORITY.values())
priorities_array.sort(reverse=True)

PERFECT_PRIORITIES = []
for pr in priorities_array:
    for _ in range(EXAM_COUNT):
        PERFECT_PRIORITIES.append(pr)

# MEDICAL_STAFF
# Medical staff represented by sequential, ordered numbers, initiating in 1!!
MEDICAL_STAFF_DISPONIBILITY = {
    1: [[420, 720], [780, 960]],  # Morning -> 300 minutes block, Afternoon -> 180 minutes block
    2: [[420, 720], [780, 960]],
    3: [[420, 720], [780, 960]],
    4: [[420, 720], [780, 960]],
    5: [[420, 720], [780, 960]],
    6: [[600, 720], [780, 1080]],  # Morning -> 120 minutes block, Afternoon -> 300 minutes block
    7: [[600, 720], [780, 1080]],
    8: [[600, 720], [780, 1080]],
    9: [[600, 720], [780, 1080]],
    10: [[600, 720], [780, 1080]],
}

MEDICAL_STAFF_COUNT = len(MEDICAL_STAFF_DISPONIBILITY)

GENERATION_NUMBER = 1



# Gene of the individuals of the population:
# medical_personnel_id: responsible for the appointment (medical staff)
# appointment_minute_of_day_start: appointment initiation in minutes, counting from the start of the day (middle night)
# appointment_minute_of_day_end: appointment end in minutes, counting from the start of the day (middle night)
# weekday: day of the week (1 - Monday, 2 - Tuesday, 3 - Thursday, ...)
# patient_id: patient ID
# room: room where the appointment will happen
class Gene:
    def __init__(self, medical_personnel_id: int, appointment_minute_of_day_start: int,
                 appointment_minute_of_day_end: int, exam_type: string, weekday: int,
                 patient_id: int, room: string):
        self.medical_personnel_id = medical_personnel_id
        self.appointment_minute_of_day_start = appointment_minute_of_day_start
        self.appointment_minute_of_day_end = appointment_minute_of_day_end
        self.exam_type = exam_type
        self.weekday = weekday
        self.patient_id = patient_id
        self.room = room

    def __str__(self):
        # return (f"Medical Staff={self.medical_personnel_id}, appointment_minute_of_day_start={self.appointment_minute_of_day_start}, "
        #        f"appointment_minute_of_day_end={self.appointment_minute_of_day_end}, exam_type={self.exam_type}, weekday={self.weekday},"
        #        f"patient_id={self.patient_id}, room={self.room}")
        return (
            f"{self.medical_personnel_id}, {self.appointment_minute_of_day_start}, "
            f"{self.appointment_minute_of_day_end}, {self.exam_type}, {self.weekday},"
            f"{self.patient_id}, {self.room}")


class Individual:
    def __init__(self, genes: [Gene], generation_born: int, operation: string, metrics: None):
        if metrics is None:
            metrics = []
        self.genes = genes
        self.generation_born = generation_born
        self.operation = operation
        self.metrics = metrics

    def __str__(self):
        for gene in self.genes:
            return gene.__str__()


# Combine the given intervals
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])

    merged = []

    for interval in intervals:
        # If the merged list is empty or the current interval doesn't overlap, add it
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # Otherwise, merge the intervals
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged


def join_intervals(intervals_1, intervals_2):
    result = []

    for interval_1 in intervals_1:
        for interval_2 in intervals_2:
            start = max(interval_1[0], interval_2[0])
            end = min(interval_1[1], interval_2[1])

            if start < end:
                result.append([start, end])
    return result


def write_individual_to_csv(individual, filename):
    # Define the header
    header = [
        "medical_personnel_id",
        "appointment_minute_of_day_start",
        "appointment_minute_of_day_end",
        "weekday",
        "patient_id",
        "room"
    ]

    # Open the CSV file for writing
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header
        writer.writerow(header)

        # Write each Gene object's attributes as a row
        for gene in individual.genes:
            writer.writerow([
                gene.medical_personnel_id,
                gene.appointment_minute_of_day_start,
                gene.appointment_minute_of_day_end,
                gene.weekday,
                gene.patient_id,
                gene.room
            ])



# get the medical staff availability for a specific exam, considering the appointments already scheduled for the specific medical staff
def get_medical_staff_availability(appointment_code, weekday, medical_personnel_id, genes):
    # get the available intervals to schedule the appointment
    medical_personnel_disponibility = MEDICAL_STAFF_DISPONIBILITY[medical_personnel_id]

    appointment_time = EXAMS_DURATION[appointment_code]

    # occupied intervals for the medicall staff, on a specific weekday
    genes_with_medical_personnel_id = list(
        filter(lambda x: x.medical_personnel_id == medical_personnel_id and x.weekday == weekday,
               genes))

    medical_personnel_occup_intervals = list(
        map(lambda x: [x.appointment_minute_of_day_start, x.appointment_minute_of_day_end],
            genes_with_medical_personnel_id))

    merged_medical_personnel_occupied_intervals = merge_intervals(medical_personnel_occup_intervals)

    # case the medical person has no occupation yet return the full availability
    if not merged_medical_personnel_occupied_intervals:
        return medical_personnel_disponibility

    medical_personnel_disponibility_minus_occupation = []
    for med_pers_disp in medical_personnel_disponibility:
        # get the ordered and merged occupations for the shift
        shift_occupations = list(filter(lambda x: x[1] <= med_pers_disp[1] and x[0] >= med_pers_disp[0],
                                        merged_medical_personnel_occupied_intervals))
        if not shift_occupations:
            medical_personnel_disponibility_minus_occupation.append(med_pers_disp)
            continue
        for index, shift_occupation in enumerate(shift_occupations):
            if index == 0:
                # is the first occupation
                if (shift_occupation[0] - med_pers_disp[0]) >= appointment_time:
                    medical_personnel_disponibility_minus_occupation.append([med_pers_disp[0], shift_occupation[0]])
            if index == len(shift_occupations) - 1:
                # is the last occupation
                if (med_pers_disp[1] - shift_occupation[1]) >= appointment_time:
                    medical_personnel_disponibility_minus_occupation.append([shift_occupation[1], med_pers_disp[1]])
                continue

            if (shift_occupations[index + 1][0] - shift_occupation[1]) >= appointment_time:
                medical_personnel_disponibility_minus_occupation.append(
                    [shift_occupation[1], shift_occupations[index + 1][0]])

    return medical_personnel_disponibility_minus_occupation


# get the availability of a room, in a individual, considering the appointments already scheduled for the room
def get_room_disponibility(appointment_code, weekday, genes):
    exam_rooms_available = ROOMS_AVAILABLE[appointment_code]

    appointment_time = EXAMS_DURATION[appointment_code]

    # shuffle the rooms on the exam_rooms_available, to iterate on them more randomly
    random.shuffle(exam_rooms_available)

    for room in exam_rooms_available:
        # occupied intervals for the room, on a specific weekday
        genes_with_room_id = list(filter(lambda x: x.room in room and x.weekday == weekday, genes))

        room_occup_intervals = list(
            map(lambda x: [x.appointment_minute_of_day_start, x.appointment_minute_of_day_end],
                genes_with_room_id))

        merged_room_occupied_intervals = merge_intervals(room_occup_intervals)

        # case the room has no occupation yet return the full availability
        if not merged_room_occupied_intervals:
            return ROOM_FULL_DISPONIBILITY, room

        room_disponibility_minus_occupation = []
        for index, occupied_interval in enumerate(merged_room_occupied_intervals):
            if index == 0:
                # is the first occupation
                if (occupied_interval[0] - ROOM_FULL_DISPONIBILITY[0][0]) >= appointment_time:
                    room_disponibility_minus_occupation.append([ROOM_FULL_DISPONIBILITY[0][0], occupied_interval[0]])
            if index == len(merged_room_occupied_intervals) - 1:
                # is the last occupation
                if (ROOM_FULL_DISPONIBILITY[0][1] - occupied_interval[1]) >= appointment_time:
                    room_disponibility_minus_occupation.append([occupied_interval[1], ROOM_FULL_DISPONIBILITY[0][1]])
                continue

            if (merged_room_occupied_intervals[index + 1][0] - occupied_interval[1]) >= appointment_time:
                room_disponibility_minus_occupation.append(
                    [occupied_interval[1], merged_room_occupied_intervals[index + 1][0]])

        return room_disponibility_minus_occupation, room


def get_appointment_disponibility(appointment_code, genes):
    ''' Identifies where the specified appointment can be scheduled

        It loops over all the weekdays and medical staff and tries to find an feasible time interval to schedule
        the passed appointment type
    '''

    appointment_disponibility = []

    appointment_time = EXAMS_DURATION[appointment_code]

    medical_staff = False
    room = False
    weekday = False

    weekdays = list(range(0, 7))
    while not appointment_disponibility and weekdays:
        # Random weekday
        weekday = weekdays[random.randint(0, len(weekdays) - 1)]
        weekdays.remove(weekday)

        all_medical_staff = list(range(1, MEDICAL_STAFF_COUNT + 1))
        # For each weekday, check the disponibility of each medical staff
        while all_medical_staff and not appointment_disponibility:
            # Random medical staff
            if len(all_medical_staff) == 1:
                medical_staff = all_medical_staff[0]
            else:
                medical_staff = all_medical_staff[random.randint(1, len(all_medical_staff) - 1)]

            all_medical_staff.remove(medical_staff)

            medical_disponibility = get_medical_staff_availability(appointment_code, weekday, medical_staff, genes)
            room_disponibility, room = get_room_disponibility(appointment_code, weekday, genes)

            medical_room_joined_disponibility = join_intervals(medical_disponibility, room_disponibility)

            if not medical_room_joined_disponibility:
                continue

            # Validates if there are space on the joined room and medical personel disponibility to schedule the desired appointment
            appointment_disponibility = list(
                filter(lambda x: x[1] - x[0] >= appointment_time, medical_room_joined_disponibility))

    # O que fazer caso nao haja disponibilidade para a consulta????????????????
    if not appointment_disponibility:
        return False, False, False, 8

    else:
        # calculate the time of the appointment
        interval_to_schedule_appointment = appointment_disponibility[
            random.randint(0, len(appointment_disponibility) - 1)]

        # generate a random multiple of 5 number, between the interval choosen previously
        appointment_init = random.choice(
            range(interval_to_schedule_appointment[0], interval_to_schedule_appointment[1] - appointment_time + 1, 5))
        appointment_end = appointment_init + appointment_time

        return [appointment_init, appointment_end], medical_staff, room, weekday


# Generates the initial population with 100 individuals
def generate_population(initial_population_size, individual_genes_size):
    # If we have 15 patients and 6 exams types, we need to schedule 15*6 appointments, thus our individual must have 90 Genes for this specific case
    population = []

    for j in range(0, initial_population_size):
        individual_genes = []

        medical_staff = False
        room = False
        weekday = False
        patient = False

        for i in range(0, individual_genes_size):
            appointment_time = []
            appointment_time_loops = 0

            # Allow 1000 iteration to find a feasible time for the  appointment... If no feadible solution is found,
            # the time of the appointment is set to [0, 0], marking this as an unfeasible solution! Hence, this needs to be considered
            # on the fitness function, so we penalize those solutions.
            #
            # WE NEED THIS TO AVOID INFINITE LOOPS, and it makes sense because we can identify appointments that could not be
            # scheduled on a certain individual
            while not appointment_time and appointment_time_loops < 1000:
                # Random appointment
                exam_type = EXAMS_TYPES[random.randint(0, EXAM_COUNT - 1)]

                # Random patient
                patient = random.randint(1, PATIENT_COUNT)

                appointment_time, medical_staff, room, weekday = get_appointment_disponibility(exam_type,
                                                                                               individual_genes)

                appointment_time_loops += 1

            if not appointment_time:
                appointment_time = [0, 0]

            gene = Gene(medical_personnel_id=medical_staff, appointment_minute_of_day_start=appointment_time[0],
                        appointment_minute_of_day_end=appointment_time[1], exam_type=exam_type, weekday=weekday,
                        patient_id=patient, room=room)
            individual_genes.append(gene)

        individual = Individual(genes=individual_genes, generation_born=1, operation="INITIAL_POPULATION", metrics=[])
        # if j == 2:
        # write_individual_to_csv(individual, "Genetic_Individual.csv")
        population.append(individual)

    return population


def calculate_patient_exams_errors(individual):
    '''
    Calculates how many times a patient has the same exam scheduled + the times a exam is missing
    for a patient + the quantity of appointments scheduled for init_time and end_time 0
    '''
    errors_count = 0

    for patient_id in PATIENT_PRIORITY:
        for exam in EXAMS_TYPES:
            filter_count_result = len(
                list(filter(lambda x: x.patient_id == patient_id and x.exam_type == exam, individual.genes)))
            if filter_count_result > 1:
                errors_count += filter_count_result - 1
            elif filter_count_result == 0:
                errors_count += 1

    normalized_value = normalize(errors_count, 0,
                                 (len(individual.genes) - 1) + (EXAM_COUNT - 1) + ((PATIENT_COUNT - 1) * EXAM_COUNT))

    return normalized_value


def calculate_overlaps(individual):
    overlaps_count = 0
    for weekday in list(range(0, 7)):
        for room in ROOMS:
            room_usages = list(filter(lambda x: x.room == room and x.weekday == weekday, individual.genes))
            if len(room_usages) > 1:

                for index, room_usage in enumerate(room_usages):
                    if index == len(room_usages) - 1:
                        break
                    appointment_time = EXAMS_DURATION[room_usage.exam_type]
                    overlaps_list = list(filter(lambda
                                                    x: room_usage.appointment_minute_of_day_start + appointment_time > x.appointment_minute_of_day_start > room_usage.appointment_minute_of_day_start,
                                                room_usages))
                    overlaps_count += len(overlaps_list)

        for medical_staff in MEDICAL_STAFF_DISPONIBILITY:
            medical_staff_usages = list(
                filter(lambda x: x.medical_personnel_id == medical_staff and x.weekday == weekday, individual.genes))
            if len(medical_staff_usages) > 1:

                for index, medical_staff_usage in enumerate(medical_staff_usages):
                    if index == len(medical_staff_usages) - 1:
                        break
                    appointment_time = EXAMS_DURATION[medical_staff_usage.exam_type]
                    overlaps_list = list(filter(lambda
                                                    x: medical_staff_usage.appointment_minute_of_day_start + appointment_time > x.appointment_minute_of_day_start > medical_staff_usage.appointment_minute_of_day_start,
                                                medical_staff_usages))
                    overlaps_count += len(overlaps_list)

    normalized_value = normalize(overlaps_count, 0, (INDIVIDUAL_GENES_COUNT * 2))

    return normalized_value


# def calculate_scheduling_overlaps(individual):
#    overlaps_count = 0
#
#    for gene in individual.genes:
#        overlaps_count = len(list(filter(lambda x: x.room == gene.room
#                                                   and (x.weekday == gene.weekday or x.medical_personnel_id ==gene.medical_personnel_id)
#                                                   and is_Time_overlaped(x.exam_type, x.appointment_minute_of_day_start,
#                                                                         gene.appointment_minute_of_day_start, gene.exam_type),
#                                         individual.genes)))
#        if overlaps_count > 1:
#            overlaps_count += overlaps_count
#
#
#    normalized_value = normalize(errors_count, 0,
#                                 (len(individual.genes) - 1) + (EXAM_COUNT - 1) + ((PATIENT_COUNT - 1) * EXAM_COUNT))
#
#    return normalized_value

def calculate_unscheduled_exams_due_lack_disponibility(individual):
    '''
    Calculates the number of exams not scheduled due to lack of disponibility
    '''

    number_exames_unscheduled = len(list(filter(lambda x: x.appointment_minute_of_day_start == 0, individual.genes)))

    return normalize(number_exames_unscheduled, 0, len(individual.genes))


def calculate_priority_assertiveness(individual, priority_min, priority_max):
    '''
    Calculates the priority assertiveness of the individual
    '''
    genes = copy.deepcopy(individual.genes)

    # order the appointments by time
    genes.sort(key=lambda x: (x.weekday, x.appointment_minute_of_day_start))

    # TODO MUDAR SORT PARA TER EMCONTA APENAS AGENDAMENTOS COM DATA
    # TODO ADICIONAR RESTANTES POR ORDEM DE PRIORIDADES

    # get the priorities of the patients from the appointments, ordered by appointment date
    ordered_priorities = list(map(lambda x: PATIENT_PRIORITY[x.patient_id], genes))

    fitness = 0
    for index, prio in enumerate(ordered_priorities):
        fitness += abs(prio - PERFECT_PRIORITIES[index])

    res = normalize(fitness, priority_min, priority_max)

    return res


def calculate_patient_preference_assertiveness(individual):
    '''
    Calculates the preference assertiveness of the individual, summing up all the differences from the patient scheduled
    exam time and his preference
    '''
    result = 0

    for appointment in individual.genes:
        patient_weekday_preference = PATIENT_PREFERENCE[appointment.patient_id][appointment.weekday]

        if patient_weekday_preference == 0:
            result += 720
        else:
            if appointment.appointment_minute_of_day_start:
                result += abs(patient_weekday_preference - appointment.appointment_minute_of_day_start)

    result_normalized = normalize(result, 0, 720 * len(individual.genes))

    return result_normalized


def calculate_fitness_function(individual):
    '''
    Calculates the fitness value of the individual

    To calculate the fitness, consider:
        - Priorities values
        - Patient time preferences
        - Individual unfeasible level (unfeasbile individuals are possible because we dont ensure that in each individual a
          patient has the correct number of exams, AND there are patients whose exam could not be scheduled, so the gene
          appointment initial and end time is set to 0)
        - Medical staff exams load distribuction
        - Unscheduled exams due to lack of disponibility

        MAXIMIZE:
            - PRIORITY WEIGHTING RESULT
        MINIMIZE:
            - UNFEASIBLE SOLUTIONS ERRORS
            - UNSCHEDULED EXAMS
            - INDIVIDUAL UNFEASIBLE LEVEL
    '''
    # the more errors we got, the worse the result is
    patient_exams_errors_penalization = calculate_patient_exams_errors(individual)

    # the more unscheduled exams we got, the worse the result is
    unscheduled_exams_penalization = calculate_unscheduled_exams_due_lack_disponibility(individual)

    # a higher output from this function indicates a better priority assertiveness
    priority_assertiveness_penalization = calculate_priority_assertiveness(individual, PRIORITY_MIN, PRIORITY_MAX)

    patient_preference_assertiveness_penalization = calculate_patient_preference_assertiveness(individual)

    overlaps_penalization = calculate_overlaps(individual)

    total_weighted_penalization = (patient_exams_errors_penalization * FITNESS_ERRORS_WEIGHT
                                   + overlaps_penalization * FITNESS_OVERLAPS_WEIGHT
                                   + unscheduled_exams_penalization * FITNESS_UNSCHEDULED_EXAMS_WEIGHT
                                   + priority_assertiveness_penalization * FITNESS_PRIORITY_WEIGHT
                                   + patient_preference_assertiveness_penalization * FITNESS_PREFERENCE_WEIGHT)

    max_penalization_pond = (FITNESS_ERRORS_WEIGHT + FITNESS_OVERLAPS_WEIGHT + FITNESS_UNSCHEDULED_EXAMS_WEIGHT +
                             FITNESS_PRIORITY_WEIGHT + FITNESS_PREFERENCE_WEIGHT)

    normalized_penalization = total_weighted_penalization / max_penalization_pond

    # print(f"Fitness calculation - {patient_exams_errors_penalization},"
    #    f"{unscheduled_exams_penalization}, "
    #    f"{priority_assertiveness_penalization}, "
    #    f"{patient_preference_assertiveness_penalization}."
    #    f"RESULT: {round(1 - normalized_penalization, 4)}")

    individual.metrics = [f"Exam errors: {patient_exams_errors_penalization} ",
                          f"Unscheduled exams: {unscheduled_exams_penalization}",
                          f"Priority assertiveness: {priority_assertiveness_penalization}",
                          f"Patient Preference: {patient_preference_assertiveness_penalization}",
                          f"Overlaps penalization: {overlaps_penalization}"]

    return 1 - normalized_penalization


def normalize(value, min_value, max_value):
    if max_value == min_value:
        return 0
    return (value - min_value) / (max_value - min_value)


def selection(population_fitness_scores):
    population_fitness_scores.sort(key=lambda x: x[1], reverse=True)
    print(f"Best individual fitness: {round(population_fitness_scores[0][1],4)} - "
          f"GENERATION:{population_fitness_scores[0][0].generation_born} "
          f"FROM:{population_fitness_scores[0][0].operation} ")

    print(f"metrics: {population_fitness_scores[0][0].metrics}")
    population = list(map(lambda x: x[0], population_fitness_scores))

    # Calculate the total fitness and relative probabilities
    total_fitness = sum(list(map(lambda x: x[1], population_fitness_scores)))
    if total_fitness == 0:
        raise ValueError("Total fitness is zero, selection cannot be performed.")

    # number of individuals to select
    num_selections = math.ceil((1 - SUBSTITUTION_RATE) * len(population_fitness_scores))


    num_pity_selections = math.ceil(num_selections * PITY_SELECTION_RATE)
    num_elitist_selections = math.ceil(num_selections - num_pity_selections)

    #selected_pity_individuals= generate_population(num_pity_selections, INDIVIDUAL_GENES_COUNT)

    # select elite
    if num_elitist_selections % 2 == 0:
        selected_elite_individuals = population[:num_elitist_selections]
    else:
        selected_elite_individuals = population[:num_elitist_selections - 1]

    selected_pity_individuals = population[-num_pity_selections:]


    #selected_elite_individuals=  selected_elite_individuals[:len(selected_elite_individuals) // 2]
    #elite_mutated = mutation(selected_elite_individuals, 1)
    #elite_mutated_fitness = []
    #for individual in elite_mutated:
     #   individual1_fitness = calculate_fitness_function(individual)
    #    elite_mutated_fitness.append((individual, individual1_fitness))

    #elite_mutated_fitness.sort(key=lambda x: x[1], reverse=True)
    #elite_mutated = list(map(lambda x: x[0], elite_mutated_fitness))
    #elite_mutated = elite_mutated[:num_pity_selections]

    #return selected_elite_individuals+ elite_mutated  + selected_pity_individuals
    return selected_elite_individuals + selected_pity_individuals


def selection_and_crossover(population):
    """
    Calculate the fitness function of each individual and call the roullete wheel selection
    """

    population_fitness_score = []
    for individual in population:
        individual1_fitness = calculate_fitness_function(individual)
        population_fitness_score.append((individual, individual1_fitness))

    selected_pop = selection(population_fitness_score)
    descendents = []
    min_descendents_count = POPULATION_SIZE - len(selected_pop)

    while len(descendents) < min_descendents_count:

        selected_parent_1_index=random.randint(0, len(selected_pop) - 1)
        selected_parent_1=selected_pop[selected_parent_1_index]


        selected_parent_2_index = random.randint(0, len(selected_pop) - 1)
        selected_parent_2 = selected_pop[selected_parent_2_index]

        crossover_result = crossover([selected_parent_1, selected_parent_2])
        if crossover_result:
            descendents += crossover_result

    descendents=descendents[:min_descendents_count]
    return selected_pop, descendents

def mutate_individual(individual):
    new_individual_genes = deepcopy(individual.genes)
    # We are mutating only a maximum of 10 genes per individual
    n_genes_to_mutate= random.randint(1, (len(individual.genes) if len(individual.genes) < 10 else 10))

    for idx in range(n_genes_to_mutate):
        i = random.randint(0, len(individual.genes) - 1)
        chromosome = random.randint(0, 5)
        random_appointment_time = random.choice(range(420, 950, 5))
        random_weekday = random.randint(0, 6)
        random_room = ROOMS[random.randint(0, len(ROOMS) - 1)]
        random_medical_staff = random.randint(0, len(MEDICAL_STAFF_DISPONIBILITY) - 1)
        random_exam = EXAMS_TYPES[random.randint(0, EXAM_COUNT - 1)]
        random_patient = random.randint(1, PATIENT_COUNT)

        new_individual_gene = Gene(
            medical_personnel_id=individual.genes[i].medical_personnel_id if chromosome != 0 else random_medical_staff,
            appointment_minute_of_day_start=individual.genes[
                i].appointment_minute_of_day_start if chromosome != 1 else random_appointment_time,
            appointment_minute_of_day_end=individual.genes[i].appointment_minute_of_day_start,
            exam_type=individual.genes[i].exam_type if chromosome != 2 else random_exam,
            weekday=individual.genes[i].weekday if chromosome != 3 else random_weekday,
            patient_id=individual.genes[i].patient_id if chromosome != 4 else random_patient,
            room=individual.genes[i].room if chromosome != 5 else random_room,
        )
        new_individual_genes[i] = new_individual_gene
        #new_individual_genes.append(new_individual_gene)

    new_individual = Individual(new_individual_genes, generation_born=GENERATION_NUMBER, operation="MUTATION",
                                    metrics=[])
    return new_individual


def mutation(individuals, mutation_rate = MUTATION_RATE):
    new_individuals = []
    for individual in individuals:
        if random.random() > mutation_rate:
            new_individuals.append(individual)
            continue

        new_individuals.append(mutate_individual(individual))
    return new_individuals


#def mutation_hardcore(individuals):
#    for individual in individuals:
#        if random.random() > MUTATION_RATE:
#            continue
#
#        for i in range(math.ceil(len(individual.genes) / 2)):
#            exam_type = EXAMS_TYPES[random.randint(0, EXAM_COUNT - 1)]
#            patient = random.randint(1, PATIENT_COUNT)
#            appointment_time, medical_staff, room, weekday = get_appointment_disponibility(exam_type, individual.genes)
#
#            random_appointment_time = random.choice(range(420, 950, 5))
#            exam_duration = EXAMS_DURATION[exam_type]
#            aggressive_mutation = random.random() > MUTATION_RATE
#            random_weekday = random.randint(0, 6)
#            random_room = ROOMS[random.randint(0, len(ROOMS) - 1)]
#            individual.genes[i] = Gene(
#                medical_personnel_id=individual.genes[i].medical_personnel_id if aggressive_mutation else medical_staff,
#                appointment_minute_of_day_start=appointment_time[0] if aggressive_mutation else random_appointment_time,
#                appointment_minute_of_day_end=appointment_time[
#                    1] if aggressive_mutation else random_appointment_time + exam_duration,
#                exam_type=exam_type,
#                weekday=weekday if aggressive_mutation else random_weekday,
#                patient_id=patient,
#                room=room if aggressive_mutation else random_room,
#            )
#            individual.generation_born = GENERATION_NUMBER
#            individual.operation = "MUTATION"
#
#    return individuals


def crossover(parents):
    """
    Crossover can generate unfeasible solutions
    """
    # Case the random number is bigger that the crossing rate, do not cross the individuals
    crossing_coin_toss = random.random()
    if crossing_coin_toss > CROSSING_RATE:
        return []

    descendents = []
    descendent1_genes = []
    descendent2_genes = []

    for parent_gene_index, gene in enumerate(parents[0].genes):
        parent1_gene = gene
        parent2_gene = parents[1].genes[parent_gene_index]

        generated_gene_1 = []
        generated_gene_2 = []
        crossover_mask_gene = [random.randint(0, 1) for _ in range(7)]
        # Cross the gene itself
        for gene_cel_index, gene_bit in enumerate(crossover_mask_gene):
            if gene_bit == 0:
                generated_gene_1.append(list(vars(parent1_gene).values())[gene_cel_index])
                generated_gene_2.append(list(vars(parent2_gene).values())[gene_cel_index])
            else:
                generated_gene_1.append(list(vars(parent2_gene).values())[gene_cel_index])
                generated_gene_2.append(list(vars(parent1_gene).values())[gene_cel_index])

        descendent1_genes.append(
            Gene(medical_personnel_id=generated_gene_1[0], appointment_minute_of_day_start=generated_gene_1[1],
                 appointment_minute_of_day_end=generated_gene_1[2], exam_type=generated_gene_1[3],
                 weekday=generated_gene_1[4],
                 patient_id=generated_gene_1[5], room=generated_gene_1[6]))
        descendent2_genes.append(
            Gene(medical_personnel_id=generated_gene_2[0], appointment_minute_of_day_start=generated_gene_2[1],
                 appointment_minute_of_day_end=generated_gene_2[2], exam_type=generated_gene_2[3],
                 weekday=generated_gene_2[4],
                 patient_id=generated_gene_2[5], room=generated_gene_2[6]))

    descendents.append(
        Individual(genes=descendent1_genes, operation="CROSSOVER", generation_born=GENERATION_NUMBER, metrics=[]))
    descendents.append(
        Individual(genes=descendent2_genes, operation="CROSSOVER", generation_born=GENERATION_NUMBER, metrics=[]))

    return descendents


def calculate_metrics(population):
    global average_fitness_values
    global best_fitness_values

    fitness = list(map(lambda x: calculate_fitness_function(x), population))
    average_population_fitness = reduce(lambda x, y: x + y, fitness, 0) / len(population)

    # print(f"fitness antes de ordenar: {fitness[0]}")
    average_fitness_values.append(average_population_fitness)
    top_individual_fitness = max(fitness)
    best_fitness_values.append(top_individual_fitness)

    return


def main():
    population = generate_population(POPULATION_SIZE, INDIVIDUAL_GENES_COUNT)
    global GENERATION_NUMBER
    print(str(population))

    GENERATION_NUMBER = 1

    while GENERATION_NUMBER < NUMBER_OF_GENERATIONS:
        print("GENERATION NUMBER: " + str(GENERATION_NUMBER))
        selected_population, descendents = selection_and_crossover(population)

        # descendents = crossover(selected_population)
        mutated = mutation(descendents)

        population = selected_population + mutated

        calculate_metrics(population)
        print(f"Average population fitness: {str(average_fitness_values[-1])}")
        GENERATION_NUMBER += 1

    fitness_pop = []
    index_best, best_fitness = 0, 0
    i = 0
    for index, individual in enumerate(population):
        fitness_pop.append(calculate_fitness_function(individual))
        print(f"INDIVIDUAL {str(i)}: fitness: {str(fitness_pop[index])}")
        if fitness_pop[index] > best_fitness:
            best_fitness = fitness_pop[index]
            index_best = index
        i += 1

    print(str(population[index_best].genes))

    mapped_schedule = parse_schedule(population[index_best].genes)
    display_schedule_table(mapped_schedule)
    generations = list(range(1, len(average_fitness_values) + 1))

    plt.plot(generations, average_fitness_values, label="Avg. Population Fitness")
    plt.plot(generations, best_fitness_values, label="Best Individual Fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("Fitness Evolution")
    plt.legend()
    plt.show()


def parse_schedule(schedules):
    """Parse the schedule string into a list of dictionaries."""
    mapped_schedule = []

    for schedule in schedules:
        entry = {
            'medical_staff': schedule.medical_personnel_id,
            'appointment_time': (schedule.appointment_minute_of_day_start, schedule.appointment_minute_of_day_end),
            'exam_type': schedule.exam_type,
            'weekday': schedule.weekday,
            'patient': schedule.patient_id,
            'room': schedule.room
        }
        mapped_schedule.append(entry)
    return mapped_schedule


import matplotlib.pyplot as plt


def display_schedule_table(schedule):
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Criar lista de registros para a tabela
    table_data = []
    for entry in schedule:
        start_hour = entry['appointment_time'][0] // 60
        start_minute = entry['appointment_time'][0] % 60
        end_hour = entry['appointment_time'][1] // 60
        end_minute = entry['appointment_time'][1] % 60

        start_time = f"{start_hour:02d}:{start_minute:02d}"
        end_time = f"{end_hour:02d}:{end_minute:02d}"

        table_data.append({
            'Weekday': weekdays[entry['weekday']],
            'Start Time': start_time,
            'End Time': end_time,
            'Patient ID': entry['patient'],
            'Exam Type': entry['exam_type'],
            'Medical Staff': entry['medical_staff'],
            'Room': entry['room']
        })

    # Criar DataFrame ordenado por dia da semana e horário de início
    df = pd.DataFrame(table_data)
    df['Weekday Order'] = df['Weekday'].apply(lambda x: weekdays.index(x))
    df['Start Time (Minutes)'] = df['Start Time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    df = df.sort_values(by=['Weekday Order', 'Start Time (Minutes)'])

    # Remover colunas auxiliares
    df = df.drop(columns=['Weekday Order', 'Start Time (Minutes)'])

    # Criar figura para o gráfico da tabela
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.5))
    ax.axis('tight')
    ax.axis('off')

    # Adicionar a tabela ao gráfico
    tbl = table.table(
        ax,
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))

    plt.show()


main()

