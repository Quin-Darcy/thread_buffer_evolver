use std::path::PathBuf;
use std::process::Command;

use rand::Rng;
use regex::Regex;

extern crate num_cpus;

use crate::config::Params;
use crate::constants::{
    INITIAL_POPULATION_SIZE, 
    TOURNAMENT_WINNERS_PERCENTAGE, 
    SELECTION_PRESSURE, 
    NUMBER_OF_GENERATIONS,
    CROSSOVER_RATE,
    MUTATION_RATE,
};


pub struct GA {
    population: Vec<(Params, f64)>,
    file_path: PathBuf,
    max_threads: usize,
    max_buffer_size: usize,
    current_best: (Params, f64),
}

impl GA {
    pub fn new(file_path: PathBuf, file_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        let max_threads = num_cpus::get();
        let max_buffer_size = file_size;
        let mut population: Vec<(Params, f64)> = Vec::with_capacity(INITIAL_POPULATION_SIZE);

        for _ in 0..INITIAL_POPULATION_SIZE {
            population.push((Params { threads: rng.gen_range(1..=max_threads), buffer_size: rng.gen_range(1..=max_buffer_size)}, 0.0));
        }

        GA { population: population, file_path: file_path, max_threads: max_threads, max_buffer_size: max_buffer_size, current_best: (Params { threads: 1, buffer_size: 1 }, 0.0) }
    }

    pub fn evolve(&mut self) -> (Params, f64) {
        for i in 0..NUMBER_OF_GENERATIONS {
            if i == 0 {
                self.evaluate();
            }

            // Perform tournament selection
            self.tournament_selection();

            // Perform crossover
            self.crossover();

            // Perform mutation
            self.mutate();

            // Evaluate all members of the population
            self.evaluate();

            // Prune the population to get it back to max capacity
            self.prune();

            println!("    Generation: {}", i);
            println!("        File Size: {}", self.max_buffer_size);
            println!("        Average Fitness: {}", self.get_average_fitness());
            println!("        Current Best: {:?}", self.get_best_individual().0);
        }

        return self.get_best_individual();
    }

    fn evaluate(&mut self) {
        // Call the command to run the binary and capture its performs metric using a single thread
        let singlethread_output = Command::new("sh")
            .arg("-c")
            .arg(format!("perf stat -e cache-misses,cache-references,instructions,cycles ./entropy {} {} {}", 1, 1, self.file_path.to_str().unwrap()))
            .output()
            .expect("failed to execute process");

        let singlethread_perf_stderr = String::from_utf8_lossy(&singlethread_output.stderr);

        // Create regex for each of the values we need to capture
        let re_cache_misses = Regex::new(r"(\d{1,3}(?:,\d{3})*)\s+cache-misses:u").unwrap();
        let re_cache_references = Regex::new(r"(\d{1,3}(?:,\d{3})*)\s+cache-references:u").unwrap();
        let re_instructions = Regex::new(r"(\d{1,3}(?:,\d{3})*)\s+instructions").unwrap();
        let re_cycles = Regex::new(r"(\d{1,3}(?:,\d{3})*)\s+cycles").unwrap();
        let re_time_elapsed = Regex::new(r"(\d+(?:\.\d+)?)\s+seconds time elapsed").unwrap();

        // Store the single-threaded execution time in its own variable
        let mut singlethread_time_elapsed: f64 = 0.0;
        let cap = re_time_elapsed.captures(&singlethread_perf_stderr);
        match cap {
            Some(caps) => {
                if let Some(matched) = caps.get(1) {
                    let number_str = matched.as_str();
                    if let Ok(number) = number_str.parse::<f64>() {
                        singlethread_time_elapsed = number;
                    }
                } 
            },
            None => {}
        };

        // Initialize the vector which will store the fitness values
        let mut fitness_values = Vec::with_capacity(self.population.len());

        // Loop through each individual in the population (number of threads, buffer size) and run the binary and evaulate its fitness 
        for individual in &self.population {
            // Only evalute those who haven't been evaluated yet, i.e., those whose fitness is 0.0
            if individual.1 == 0.0 {
                // Run the binary passing in the indiviual's components as the arguments to the binary
                let output = Command::new("sh")
                    .arg("-c")
                    .arg(format!("perf stat -e cache-misses,cache-references,instructions,cycles ./entropy {} {} {}", individual.0.threads, individual.0.buffer_size, self.file_path.to_str().unwrap()))
                    .output()
                    .expect("failed to execute process");
                
                let perf_stderr = String::from_utf8_lossy(&output.stderr);

                // Create vector to store all the captured metrics
                let mut metrics: Vec<f64> = Vec::new();

                // Function to parse and add metric value to the 'values' vector
                let mut parse_and_add = |regex: &Regex, text: &str| {
                    if let Some(caps) = regex.captures(text) {
                        if let Some(matched) = caps.get(1) {
                            let number_str = matched.as_str().replace(",", "");
                            if let Ok(number) = number_str.parse::<f64>() {
                                metrics.push(number);
                            }
                        }
                    }
                };

                // Parse each metric
                parse_and_add(&re_cache_misses, &perf_stderr);
                parse_and_add(&re_cache_references, &perf_stderr);
                parse_and_add(&re_instructions, &perf_stderr);
                parse_and_add(&re_cycles, &perf_stderr);
                parse_and_add(&re_time_elapsed, &perf_stderr);

                /*
                    metrics[0] := cache-misses
                    metrics[1] := cache-references
                    metrics[2] := instructions
                    metrics[3] := cycles
                    metrics[4] := time elapsed
                 */

                // Compute fitness based on extracted metrics
                let fitness = self.compute_fitness(
                    individual.0.threads, 
                    metrics[0] as u64, 
                    metrics[1] as u64, 
                    metrics[2] as u64, 
                    metrics[3] as u64, 
                    metrics[4], 
                    singlethread_time_elapsed
                );

                fitness_values.push(fitness);
            } else {
                fitness_values.push(individual.1);
            }
        }

        // Update each individual with their newly computed fitnesses
        for (individual, &fitness) in self.population.iter_mut().zip(fitness_values.iter()) {
            individual.1 = fitness;

            if individual.0.threads == 0 {
                individual.0.threads = 1;
            }

            if individual.0.buffer_size == 0 {
                individual.0.buffer_size = 1;
            }

            if fitness > self.current_best.1 {
                self.current_best = individual.clone();
            }
        }
        
    }

    fn tournament_selection(&mut self) {
        let mut rng = rand::thread_rng();
        let population_size = self.population.len();
        let number_of_winners = (population_size as f32 * TOURNAMENT_WINNERS_PERCENTAGE).ceil() as usize;
        let mut winners: Vec<(Params, f64)> = Vec::with_capacity(number_of_winners);

        // Break population into a queue of batches where each batch will compete in a tournament
        let queue = self.get_queue(number_of_winners, population_size);

        if queue.is_empty() {
            println!("queue is empty");
            panic!();
        }

        winners.push(self.current_best.clone());

        // Perform tournament selection on each batch and add the winner to the winners vector
        for i in 0..number_of_winners {
            let winner_index = if rng.gen::<f32>() > SELECTION_PRESSURE {
                rng.gen_range(0..queue[i].len())
            } else {
                let mut fittest_index = 0;
                let mut max_fitness = f64::MIN;

                for j in 0..queue[i].len() {
                    let fitness = self.population[j].1;
                    if fitness > max_fitness {
                        max_fitness = fitness;
                        fittest_index = j;
                    }
                }

                fittest_index
            };

            if self.population[winner_index].0.threads == self.current_best.0.threads && self.population[winner_index].0.buffer_size == self.current_best.0.buffer_size {
                continue;
            }

            winners.push(self.population[winner_index].clone())
        }

        self.population = winners;
    } 

    fn crossover(&mut self) {
        let mut rng = rand::thread_rng();
        let mut new_population: Vec<(Params, f64)> = Vec::new();
        let total_fitness: f64 = self.population.iter().map(|x| x.1).sum();

        new_population.push(self.current_best.clone());
    
        for _ in 0..self.population.len() {
            if rng.gen::<f32>() < CROSSOVER_RATE {
                let index = rng.gen_range(0..self.population.len());

                if self.population[index].0.threads == self.current_best.0.threads && self.population[index].0.buffer_size == self.current_best.0.buffer_size {
                    continue;
                }

                new_population.push(self.population[index].clone());
            } else {
                let mate1 = self.roulette_wheel_selection(&self.population, total_fitness, &mut rng);
                let mate2 = self.roulette_wheel_selection(&self.population, total_fitness, &mut rng);
    
                let child1_threads = mate1.threads.min(self.max_threads);
                let child1_buffer_size = mate2.buffer_size.min(self.max_buffer_size);
                let child1 = Params { threads: child1_threads, buffer_size: child1_buffer_size };

                let child2_threads = mate2.threads.min(self.max_threads);
                let child2_buffer_size = mate1.buffer_size.min(self.max_buffer_size);
                let child2 = Params { threads: child2_threads, buffer_size: child2_buffer_size };

                new_population.push((child1, 0.0));
                new_population.push((child2, 0.0));
            }
        }
    
        self.population = new_population;
    }
    
    
    fn roulette_wheel_selection(&self, population: &Vec<(Params, f64)>, total_fitness: f64, rng: &mut rand::rngs::ThreadRng) -> Params {
        let mut slice = rng.gen::<f64>() * total_fitness;
        let mut cumulative = 0.0;
        for individual in population {
            cumulative += individual.1;
            if cumulative >= slice {
                return individual.0;  // Return a copy of Params
            }
        }
        population.last().unwrap().0  // Return a copy of Params
    }
    fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        
        for i in 0..self.population.len() {
            if rng.gen::<f32>() < MUTATION_RATE {
                if self.population[i].0.threads == self.current_best.0.threads && self.population[i].0.buffer_size == self.current_best.0.buffer_size {
                    continue;
                }

                let mut current_threads = self.population[i].0.threads;
                let mut current_buffer_size = self.population[i].0.buffer_size;
                let choice = rng.gen::<f32>();
    
                if choice < 0.25 {
                    current_threads = current_threads.saturating_add(1).min(self.max_threads);
                } else if choice >= 0.25 && choice < 0.50 {
                    current_threads = current_threads.saturating_sub(1).max(1);
                } else if choice >= 0.50 && choice < 0.75 {
                    let ratio = rng.gen_range(0.1..=0.8);
                    let change = (ratio * current_buffer_size as f32).ceil() as usize;
                    current_buffer_size = current_buffer_size.saturating_add(change).min(self.max_buffer_size);
                } else {
                    let ratio = rng.gen_range(0.1..=0.8);
                    let change = (ratio * current_buffer_size as f32).ceil() as usize;
                    current_buffer_size = current_buffer_size.saturating_sub(change).max(1);
                }
    
                self.population[i].0.threads = current_threads;
                self.population[i].0.buffer_size = current_buffer_size;
            }
        }
    }
    
    fn get_queue(&self, number_of_winners: usize, population_size: usize) -> Vec<Vec<(Params, f64)>> {
        let batch_size = population_size / number_of_winners;
        let remainder = population_size % number_of_winners;

        let mut queue: Vec<Vec<(Params, f64)>> = Vec::with_capacity(number_of_winners);

        let mut current_index = 0;
        for i in 0..number_of_winners {
            let mut current_batch_size = batch_size;
            if i < remainder {
                current_batch_size += 1;
            }

            let end_index = current_index + current_batch_size;
            let batch: Vec<(Params, f64)> = self.population[current_index..end_index]
                .iter()
                .map(|x| x.clone())
                .collect();

            queue.push(batch);

            current_index = end_index;
        }

        return queue;
    }

    pub fn prune(&mut self) {
        // Sort population by fitness, but keep unevaluated individuals (fitness == 0.0) at the end.
        self.population.sort_by(|a, b| {
            if a.1 == 0.0 {
                std::cmp::Ordering::Greater
            } else if b.1 == 0.0 {
                std::cmp::Ordering::Less
            } else {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        // Reduce the population size down to INITIAL_POPULATION_SIZE, removing the lowest fitness individuals.
        while self.population.len() > INITIAL_POPULATION_SIZE {
            self.population.pop();
        }
    }

    fn get_best_individual(&self) -> (Params, f64) {
        self.current_best
    }

    fn compute_fitness(
        &self,
        num_of_threads: usize, 
        cache_misses: u64, 
        cache_references: u64, 
        _instructions: u64, 
        _cycles: u64, 
        time_elapsed: f64, 
        singlethread_time_elapsed: f64
    ) -> f64 {
        // Calculate resource efficiency
        let resource_efficiency = if cache_references > 0 {
            (cache_references as f64 - cache_misses as f64) / cache_references as f64
        } else {
            0.0
        };
    
        // Calculate parallelization efficiency
        let speedup = singlethread_time_elapsed / time_elapsed;
        let parallelization_efficiency = speedup / num_of_threads as f64;
    
        // Normalize the metrics
        let normalized_time = 1.0 / (1.0 + time_elapsed); // Normalize to a range of 0 to 1
        let normalized_resource_efficiency = resource_efficiency; // Already between 0 and 1
        let normalized_parallelization_efficiency = 1.0 / (1.0 + (1.0 / parallelization_efficiency)); // Normalize to a range of 0 to 1
    
        // Weighted Sum
        const WEIGHT_TIME: f64 = 0.2;
        const WEIGHT_PARALLELIZATION: f64 = 0.5;
        const WEIGHT_RESOURCE: f64 = 0.2;
    
        let fitness = WEIGHT_TIME * normalized_time 
            + WEIGHT_PARALLELIZATION * normalized_parallelization_efficiency 
            + WEIGHT_RESOURCE * normalized_resource_efficiency;
    
        fitness.min(1.0).max(0.0) // Ensure fitness is within 0 to 1
    }
    

    fn get_average_fitness(&self) -> f64 {
        let population_size = self.population.len();
        if population_size == 0 {
            return 0.0;
        }
    
        let fitness_sum: f64 = self.population.iter()
            .map(|(_, fitness)| {
                if fitness.is_infinite() {
                    1.0
                } else {
                    *fitness
                }
            })
            .sum();
    
        fitness_sum / population_size as f64
    }
    
    
}