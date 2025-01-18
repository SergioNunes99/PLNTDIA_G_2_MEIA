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
import math
import random
import string
from functools import reduce

#============================================================================================================
#Algorithm optimization parameters

CROSSING_RATE = 0.7
SUBSTITUTION_RATE = 0.4
MUTATION_RATE = 0.05

MIN_POPULATION_SIZE = 90
MAX_POPULATION_SIZE = 120

#============================================================================================================

INITIAL_POPULATION_SIZE = 100
NUMBER_OF_GENERATIONS = 200

#EXAMS
EXAMS_TYPES = ['BT','ECG','EG','AC','ECO','RX']

EXAMS_DURATION = {'BT': 10,
        'ECG': 10,
        'EG': 60,
        'AC': 20,
        'ECO': 30,
        'RX': 10}

EXAM_COUNT = len(EXAMS_TYPES)

#================================ FITNESS =======================================================================
PRIORITY_BEST_CASE = 0
PRIORITY_WORST_CASE= 0

FITNESS_ERRORS_WEIGHT = 0.3
FITNESS_UNSCHEDULED_EXAMS_WEIGHT = 0.3
FITNESS_PRIORITY_WEIGHT = 0.3
FITNESS_PREFERENCE_WEIGHT = 0.1

#================================================================================================================

#ROOMS
rooms = ['room1', 'room2', 'room3', 'room4', 'room5', 'room6', 'room7']

ROOMS_AVAILABLE = {'BT': [rooms[0], rooms[3]],
'ECG': [rooms[1], rooms[5]],
'EG': [rooms[2], rooms[6]],
'AC': [rooms[0]],
'ECO': [rooms[4], rooms[3]],
'RX': [rooms[1], rooms[2]]}

ROOM_FULL_DISPONIBILITY = [[0, 1440]]

#PATIENTS
#Patients represented by sequential, ordered numbers, initiating in 1!!
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
}

PATIENT_COUNT = len(PATIENT_PREFERENCE)

#MEDICAL_STAFF
#Medical staff represented by sequential, ordered numbers, initiating in 1!!
MEDICAL_STAFF_DISPONIBILITY = {
1: [[420, 720], [780, 960]], # Morning -> 300 minutes block, Afternoon -> 180 minutes block
2: [[420, 720], [780, 960]],
3: [[420, 720], [780, 960]],
4: [[420, 720], [780, 960]],
5: [[420, 720], [780, 960]],
6: [[600, 720], [780, 1080]], # Morning -> 120 minutes block, Afternoon -> 300 minutes block
7: [[600, 720], [780, 1080]],
8: [[600, 720], [780, 1080]],
9: [[600, 720], [780, 1080]],
10: [[600, 720], [780, 1080]],
}

MEDICAL_STAFF_COUNT = len(MEDICAL_STAFF_DISPONIBILITY)

INDIVIDUAL_GENES_SIZE = PATIENT_COUNT * EXAM_COUNT

# Gene of the individuals of the population:
# medical_personnel_id: responsible for the appointment (medical staff)
# appointment_minute_of_day_start: appointment initiation in minutes, counting from the start of the day (middle night)
# appointment_minute_of_day_end: appointment end in minutes, counting from the start of the day (middle night)
# weekday: day of the week (1 - Monday, 2 - Tuesday, 3 - Thursday, ...)
# pacient_id: pacient ID
# room: room where the appointent will happen
class Gene:
    def __init__(self, medical_personnel_id: int, appointment_minute_of_day_start: int, appointment_minute_of_day_end: int, exam_type: string, weekday: int,
                patient_id: int, room: string):
        self.medical_personnel_id = medical_personnel_id
        self.appointment_minute_of_day_start = appointment_minute_of_day_start
        self.appointment_minute_of_day_end = appointment_minute_of_day_end
        self.exam_type = exam_type
        self.weekday = weekday
        self.patient_id = patient_id
        self.room = room

    def __str__(self):
        return (f"Medical Staff={self.medical_personnel_id}, appointment_minute_of_day_start={self.appointment_minute_of_day_start}, "
                f"appointment_minute_of_day_end={self.appointment_minute_of_day_end}, exam_type={self.exam_type}, weekday={self.weekday},"
                f"patient_id={self.patient_id}, room={self.room}")

class Individual:
    def __init__(self, genes: [Gene]):
        self.genes = genes


#Combine the given intervals
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



#AVOID UNFEASIBLE SOLUTIONS!!
#get the medical staff availability for a specific exam, considering the appointments already scheduled for the specific medical staff
def get_medical_staff_disponibility(appointment_code, weekday, medical_personnel_id, genes):
    # get the available intervals to schedule the appointment
    medical_personnel_disponibility = MEDICAL_STAFF_DISPONIBILITY[medical_personnel_id]

    appointment_time = EXAMS_DURATION[appointment_code]

    #occupied intervals for the medicall staff, on a specific weekday
    genes_with_medical_personnel_id = list(filter(lambda x: x.medical_personnel_id == medical_personnel_id and x.weekday == weekday,
                                                  genes))

    medical_personnel_occup_intervals = list(map(lambda x: [x.appointment_minute_of_day_start, x.appointment_minute_of_day_end],
                                            genes_with_medical_personnel_id))

    merged_medical_personnel_occupied_intervals = merge_intervals(medical_personnel_occup_intervals)

    #case the medical person has no occupation yet return the full availability
    if not merged_medical_personnel_occupied_intervals:
        return medical_personnel_disponibility

    medical_personnel_disponibility_minus_occupation = []
    for med_pers_disp in medical_personnel_disponibility:
        #get the ordered and merged occupations for the shift
        shift_occupations = list(filter(lambda x: x[1] <= med_pers_disp[1] and x[0] >= med_pers_disp[0],
                                     merged_medical_personnel_occupied_intervals))
        if not shift_occupations:
            medical_personnel_disponibility_minus_occupation.append(med_pers_disp)
            continue
        for index, shift_occupation in enumerate(shift_occupations):
            if index == 0:
                #is the first occupation
                if (shift_occupation[0] - med_pers_disp[0]) >= appointment_time:
                    medical_personnel_disponibility_minus_occupation.append([med_pers_disp[0], shift_occupation[0]])
            if index == len(shift_occupations) - 1:
                # is the last occupation
                if (med_pers_disp[1] - shift_occupation[1]) >= appointment_time:
                    medical_personnel_disponibility_minus_occupation.append([shift_occupation[1], med_pers_disp[1]])
                continue

            if (shift_occupations[index + 1][0] - shift_occupation[1]) >= appointment_time:
                medical_personnel_disponibility_minus_occupation.append([shift_occupation[1], shift_occupations[index + 1][0]])

    return medical_personnel_disponibility_minus_occupation

# get the availability of a room, in a individual, considering the appointments already scheduled for the room
def get_room_disponibility(appointment_code, weekday, genes):
    exam_rooms_available = ROOMS_AVAILABLE[appointment_code]

    appointment_time = EXAMS_DURATION[appointment_code]

    #shuffle the rooms on the exam_rooms_available, to iterate on them more randomly
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

        all_medical_staff = list(range(1, MEDICAL_STAFF_COUNT  + 1))
        #For each weekday, check the disponibility of each medical staff
        while all_medical_staff and not appointment_disponibility:
            # Random medical staff
            medical_staff = all_medical_staff[random.randint(1, len(all_medical_staff) - 1)]
            all_medical_staff.remove(medical_staff)

            medical_disponibility = get_medical_staff_disponibility(appointment_code, weekday, medical_staff, genes)
            room_disponibility, room = get_room_disponibility(appointment_code, weekday, genes)

            medical_room_joined_disponibility = join_intervals(medical_disponibility, room_disponibility)

            if not medical_room_joined_disponibility:
                continue

            #Validates if there are space on the joined room and medical personel disponibility to schedule the desired appointment
            appointment_disponibility = list(filter(lambda x: x[1] - x[0] >=  appointment_time, medical_room_joined_disponibility))

    #O que fazer caso nao haja disponibilidade para a consulta????????????????
    if not appointment_disponibility:
        return False, False, False, False

    else:
        #calculate the time of the appointment
        interval_to_schedule_appointment = appointment_disponibility[random.randint(0, len(appointment_disponibility) - 1)]

        #generate a random multiple of 5 number, between the interval choosen previously
        appointment_init = random.choice(range(interval_to_schedule_appointment[0], interval_to_schedule_appointment[1] - appointment_time + 1, 5))
        appointment_end = appointment_init + appointment_time

        return [appointment_init, appointment_end], medical_staff, room, weekday


#Generates the initial population with 100 individuals
def generate_population():
    #If we have 15 patients and 6 exams types, we need to schedule 15*6 appointments, thus our individual must have 90 Genes for this specific case
    population = []

    for j in range(0, INITIAL_POPULATION_SIZE):
        individual_genes = []

        medical_staff = False
        room = False
        weekday = False
        patient = False

        for i in range(0, INDIVIDUAL_GENES_SIZE):
            appointment_time = []
            appointment_time_loops = 0

            #Allow 1000 iteration to find a feasible time for the  appointment... If no feadible solution is found,
            #the time of the appointment is set to [0, 0], marking this as an unfeasible solution! Hence, this needs to be considered
            #on the fitness function, so we penalize those solutions.
            #
            # WE NEED THIS TO AVOID INFINITE LOOPS, and it makes sense because we can identify appointments that could not be
            #scheduled on a certain individual
            while not appointment_time and appointment_time_loops < 1000:
                #Random appointment
                exam_type = EXAMS_TYPES[random.randint(0, EXAM_COUNT - 1)]

                #Random patient
                patient = random.randint(1, PATIENT_COUNT)

                appointment_time, medical_staff, room, weekday = get_appointment_disponibility(exam_type, individual_genes)

                appointment_time_loops += 1

            if not appointment_time:
                appointment_time = [0, 0]

            gene = Gene(medical_personnel_id=medical_staff, appointment_minute_of_day_start=appointment_time[0],
                        appointment_minute_of_day_end=appointment_time[1], exam_type=exam_type, weekday=weekday, patient_id=patient, room=room)
            individual_genes.append(gene)

        individual = Individual(genes=individual_genes)
        #if j == 2:
            #write_individual_to_csv(individual, "Genetic_Individual.csv")
        population.append(individual)

    return population

def calculate_patient_exams_errors(individual):
    '''
    Calculates how many times a patient has the same exam scheduled + the times a exam is missing
    for a patient + the quantity of appointments scheduled for init_time and end_time 0
    '''
    errors_count = 0

    for patient in range(0, PATIENT_COUNT):
        for exam in  EXAMS_TYPES:
            filter_count_result = len(list(filter(lambda x: x.patient_id == patient and x.exam_type == exam, individual.genes)))
            if filter_count_result > 1:
                errors_count += filter_count_result
            elif filter_count_result == 0:
                errors_count += 1

    normalized_value = normalize(errors_count, 0, PATIENT_COUNT* len(individual.genes))

    return normalized_value

def calculate_unscheduled_exams_due_lack_disponibility(individual):
    '''
    Calculates the number of exams not scheduled due to lack of disponibility
    '''

    number_exames_unscheduled= len(list(filter(lambda x: x.appointment_minute_of_day_start == 0, individual.genes)))

    return normalize(number_exames_unscheduled, 0, PATIENT_COUNT* EXAM_COUNT)

def calculate_priority_assertiveness(individual):
    '''
    Calculates the priority assertiveness of the individual
    '''

    genes = copy.deepcopy(individual.genes)

    #order the appointments by time
    genes.sort(key=lambda x: (x.weekday, x.appointment_minute_of_day_start))

    #get the priorities of the patients from the appointments, ordered by appointment date
    ordered_priorities = list(map(lambda x: PATIENT_PRIORITY[x.patient_id], genes))

    #for a better suit regarding the priorities, as the index of the array grows, the priorities must grow too, thus,
    #lower priorities must be in the first indexes of the array, and the higher values at the end of the array
    #
    #there are PATIENT_COUNT*EXAM_COUNT genes, hence PATIENT_COUNT*EXAM_COUNT indexes on the ordered_priorities array
    #
    #to ponderate the result, we multiply the value of each priority by their index on the array, with this, the best
    #result is the highest, because is the one with the lower priorities multipling by the lower indexes and the highest
    #priorities multiplying by the highest indexes

    fitness = reduce(lambda a, b: a + b, [x * index for index, x in enumerate(ordered_priorities) ])

    res = 1- normalize(fitness, PRIORITY_BEST_CASE, PRIORITY_WORST_CASE)

    if res > 1:
        res = 1

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
    #the more errors we got, the worse the result is
    patient_exams_errors_penalization = calculate_patient_exams_errors(individual)

    # the more unscheduled exams we got, the worse the result is
    unscheduled_exams_penalization = calculate_unscheduled_exams_due_lack_disponibility(individual)

    #a higher output from this function indicates a better priority assertiveness
    priority_assertiveness_penalization = calculate_priority_assertiveness(individual)

    patient_preference_assertiveness_penalization = calculate_patient_preference_assertiveness(individual)

    #TODO:: Here we can also include the weighting regarding the medical staff distribution
    #medical_staff_distribution_assertiveness = calculate_medical_staff_distribution_assertiveness(individual)



    total_weighted_penalization = (patient_exams_errors_penalization * FITNESS_ERRORS_WEIGHT
                                    + unscheduled_exams_penalization * FITNESS_UNSCHEDULED_EXAMS_WEIGHT
                                    + priority_assertiveness_penalization * FITNESS_PRIORITY_WEIGHT
                                    +patient_preference_assertiveness_penalization * FITNESS_PREFERENCE_WEIGHT)

    max_penalization_pond = (FITNESS_ERRORS_WEIGHT+ FITNESS_UNSCHEDULED_EXAMS_WEIGHT+
                             FITNESS_PRIORITY_WEIGHT + FITNESS_PREFERENCE_WEIGHT)

    normalized_penalization = total_weighted_penalization/max_penalization_pond

    #print(f"Fitness calculation - err*werr: {patient_exams_errors_penalization * FITNESS_ERRORS_WEIGHT},"
    #      f" us*wus: {unscheduled_exams_penalization * FITNESS_UNSCHEDULED_EXAMS_WEIGHT}, "
    #      f"pr*wpr: {priority_assertiveness_penalization * FITNESS_PRIORITY_WEIGHT}, "
    #      f"pre*wpre: {patient_preference_assertiveness_penalization * FITNESS_PREFERENCE_WEIGHT}."
    #      f"total_weighted_penalization: {total_weighted_penalization:.2f}, ,"
    #      f"normalized_penalization : {normalized_penalization},"
    #      f"RESULT: {1 - normalized_penalization}")

    return round(1 - normalized_penalization, 2)


def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

def selection(population_fitness_scores):
    population = list(map(lambda x: x[0], population_fitness_scores))
    fitness_scores = list(map(lambda x: x[1], population_fitness_scores))

    # Calculate the total fitness and relative probabilities
    total_fitness = sum(list(map(lambda x: x[1], population_fitness_scores)))
    if total_fitness == 0:
        raise ValueError("Total fitness is zero, selection cannot be performed.")

    #probabilities = [score / total_fitness for score in fitness_scores]

    #number of individuals to select
    num_selections = math.ceil((1 - SUBSTITUTION_RATE) * len(population_fitness_scores))

    #case the population size increases or decreases too much, set it to the default
    if num_selections < MIN_POPULATION_SIZE or num_selections > MAX_POPULATION_SIZE:
        num_selections = math.ceil((1 - SUBSTITUTION_RATE) * INITIAL_POPULATION_SIZE)

    # Perform roulette wheel selection
    #selected_individuals = []
    #individuals_to_crossover = []
    #for _ in range(int(num_selections)):
    #    pick = random.random()
    #    cumulative_probability = 0.0
    #    for individual, probability in zip(population, probabilities):
    #        cumulative_probability += probability
    #        if pick <= cumulative_probability:
    #            selected_individuals.append(individual)
    #            break
    #    #For each two individuals selected, we expose them to the crossover
    #    if len(individuals_to_crossover) == 2:
    #        descendents.append(crossover(individuals_to_crossover))
    #        individuals_to_crossover = []

    if math.ceil(num_selections) % 2 == 0:
        selected_individuals = population[:num_selections]
    else:
        selected_individuals = population[:num_selections - 1]

    return selected_individuals


def selection_and_crossover(population):
    """
    Calculate the fitness function of each individual and call the roullete wheel selection
    """

    population_fitness_score = []
    for individual in population:
        individual1_fitness = calculate_fitness_function(individual)

        population_fitness_score.append((individual, individual1_fitness))

    population_fitness_score.sort(key=lambda x: x[1])
    top_individual = population_fitness_score.pop(0)
    print("TOP INDIVIDUAL: SCORE: "+ str(top_individual[1]* 100) + "%")

    selected_pop = selection(population_fitness_score)
    descendents = []
    for index in range(0, len(selected_pop) - 1, 2):
        crossover_result = crossover([selected_pop[index], selected_pop[index + 1]])
        if crossover_result:
            descendents += crossover_result

    return [top_individual[0]] + selected_pop, descendents

# def mutation(individuals):
#     """
#     To mutate the individual, we will pick a gene randomly and mutate the assign random values to all the genes
#     internal values
#     """
#
#     selected_individual_index = random.randint(0, len(individuals) - 1)
#     selected_individual = individuals[selected_individual_index]
#
#     selected_gene_index = random.randint(0, len(selected_individual.genes) - 1)
#
#     exam_type = EXAMS_TYPES[random.randint(0, EXAM_COUNT - 1)]
#     patient = random.randint(1, PATIENT_COUNT)
#     appointment_time, medical_staff, room, weekday = get_appointment_disponibility(exam_type, selected_individual.genes)
#
#     gene = Gene(medical_personnel_id=medical_staff, appointment_minute_of_day_start=appointment_time[0],
#                 appointment_minute_of_day_end=appointment_time[1], exam_type=exam_type, weekday=weekday,
#                 patient_id=patient, room=room)
#
#     # Assign the new gene to the selected individual's genes
#     selected_individual.genes[selected_gene_index] = gene
#
#     # Replace the individual back in the population
#     individuals[selected_individual_index] = selected_individual
#
#     return individuals


def mutation(individuals):
    """
    To mutate the individual, we will assign random values to all the genes
    internal values
    """

    #selected_individual_index = random.randint(0, len(individuals) - 1)
    #selected_individual = individuals[selected_individual_index]
    new_genes = []

    for index, individual in enumerate(individuals):
        rand = random.random()
        if rand > MUTATION_RATE:
            continue
        for i in range(len(individual.genes)):

            exam_type = EXAMS_TYPES[random.randint(0, EXAM_COUNT - 1)]
            patient = random.randint(1, PATIENT_COUNT)
            appointment_time, medical_staff, room, weekday = get_appointment_disponibility(exam_type, individual.genes)

            gene = Gene(medical_personnel_id=medical_staff, appointment_minute_of_day_start=appointment_time[0],
                    appointment_minute_of_day_end=appointment_time[1], exam_type=exam_type, weekday=weekday,
                    patient_id=patient, room=room)

            # Assign the new gene to the selected individual's genes
            new_genes.append(gene)

        individual.genes = new_genes
        individuals[index] = individual

        # Replace the individual back in the population
        #individuals[selected_individual_index] = selected_individual

    return individuals

def crossover(parents):
    """
    Crossover can generate unfeasible solutions

    For the crossover, our point of cut will be the middle of the individuals
    """

    #Case the random number is bigger that the crossing rate, do not cross the individuals
    crossing_coin_toss = random.random()
    if crossing_coin_toss > CROSSING_RATE:
        return []

    descendents = []

    point_of_cut_index = int(INDIVIDUAL_GENES_SIZE/2)

    parent1_part1 = parents[0].genes[0:point_of_cut_index]
    parent1_part2 = parents[0].genes[point_of_cut_index:]

    parent2_part1 = parents[1].genes[0:point_of_cut_index]
    parent2_part2 = parents[1].genes[point_of_cut_index:]

    descendents.append(Individual(genes = parent1_part1 + parent2_part2))
    descendents.append(Individual(genes = parent2_part1 + parent1_part2))

    return descendents

def calculate_helper_values(population):
    global PRIORITY_BEST_CASE
    global PRIORITY_WORST_CASE

    genes =copy.deepcopy(population[0].genes)

    priorities = []

    for gene in genes:
        priorities.append(PATIENT_PRIORITY[gene.patient_id])

    priorities.sort()
    best_scenario = reduce(lambda a, b: a + b, [x * index for index, x in enumerate(priorities)])
    priorities.sort(reverse=True)

    worst_scenario = reduce(lambda a, b: a + b, [x * index for index, x in enumerate(priorities)])

    PRIORITY_BEST_CASE= best_scenario
    PRIORITY_WORST_CASE= worst_scenario

def main():

    #TODO: DEFINIR
    #    - TAXA DE CRUZAMENTO
    #    - TAXA DE MUTAÇÃO
    #    - TAXA DE SUBSTITUIÇÃO
    #    - CRITÉRIO DE PARAGEM

    population = generate_population()

    calculate_helper_values(population)
    print(population)

    generation_number=1

    while generation_number < NUMBER_OF_GENERATIONS:
        print("GENERATION NUMBER: " +str(generation_number))
        selected_population, descendents = selection_and_crossover(population)
        print("SELECTED POPULATION COUNT: " + str(len(selected_population)))
        #descendents = crossover(selected_population)
        mutated=mutation(descendents)

        population = selected_population + mutated
        print("END POPULATION COUNT: " + str(len(population)))

        generation_number += 1



    i=0
    for individual in population:
        print(f"INDIVIDUAL {str(i)}: fitness: {str(calculate_fitness_function(individual))}")
        i += 1

main()