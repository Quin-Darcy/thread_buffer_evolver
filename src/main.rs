use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use rand::Rng;


mod constants;
mod config;
mod ga;

use constants::MIN_FITNESS_THRESHOLD;
use config::{FileSizeRange, ParamsAndFitness, Params};
use ga::GA;


fn print_optimal_params(optimal_params: &HashMap<FileSizeRange, ParamsAndFitness>) {
    println!("{:=^80}", " Optimal Parameters "); // Prints a title with padding
    for (range, params) in optimal_params {
        println!("{:?}", range); // Assuming FileSizeRange implements Debug or Display
        println!("  Threads: {}", params.params.threads);
        println!("  Buffer Size: {}", params.params.buffer_size);
        println!("  Fitness: {:.2}", params.fitness); // Formats fitness to two decimal places
        println!("{:-^80}", ""); // Prints a separator
    }
}

fn initialize_ranges() -> HashMap<FileSizeRange, ParamsAndFitness> {
    let initial_value = ParamsAndFitness { params: Params { threads: 0, buffer_size: 0 }, fitness: 0.0 };

    let mut params_map: HashMap<FileSizeRange, ParamsAndFitness> = HashMap::new();

    let ranges = [
        FileSizeRange { min_size: 1, max_size: 1000 },
        FileSizeRange { min_size: 1001, max_size: 10000 },
        FileSizeRange { min_size: 10001, max_size: 100000 },
        FileSizeRange { min_size: 100001, max_size: 1000000 },
        FileSizeRange { min_size: 1000001, max_size: 10000000 },
        FileSizeRange { min_size: 10000001, max_size: 1000000000 },
    ];

    for range in ranges {
        params_map.insert(range, initial_value.clone());
    }

    return params_map;
}

fn load_test_files() -> Vec<PathBuf> {
    let test_files_dir = Path::new("test_files");

    let mut file_paths = Vec::new();

    if let Ok(entries) = fs::read_dir(test_files_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_file() {
                file_paths.push(path);
            }
        }
    } else {
        eprintln!("Failed to read the test_files directory.");
    }

    file_paths
}

fn all_ranges_optimized(optimal_params: &HashMap<FileSizeRange, ParamsAndFitness>) -> bool {
    for (_, &value) in optimal_params {
        if value.fitness < MIN_FITNESS_THRESHOLD {
            return false;
        }
    }

    return true;
}

fn select_random_file(file_paths: &Vec<PathBuf>) -> PathBuf {
    let mut rng = rand::thread_rng();

    let path_index: usize = rng.gen_range(0..file_paths.len());
    return file_paths[path_index].clone();
}

fn get_file_size(file_path: &PathBuf) -> usize {
    fs::metadata(file_path).unwrap().len() as usize
}

fn identify_range(optimal_params: &HashMap<FileSizeRange, ParamsAndFitness>, file_size: usize) -> FileSizeRange {
    for (&key, _) in optimal_params {
        if file_size >= key.min_size && file_size <= key.max_size {
            return key.clone();
        }
    }

    return FileSizeRange { min_size: 10000001, max_size: 1000000000 }
}


fn main() -> io::Result<()> {
    let mut optimal_params: HashMap<FileSizeRange, ParamsAndFitness> = initialize_ranges();
    let test_file_paths = load_test_files();

    let num_files = test_file_paths.len();

    for i in 0..21 {
        println!("Round: {}", i);
        // Get the file in order, wrapping around if i exceeds the number of files
        let file_index = i % num_files;
        let file_path = &test_file_paths[file_index];
        let file_size = get_file_size(&file_path);
        let file_range = identify_range(&optimal_params, file_size);

        let mut g = GA::new(file_path.clone(), file_size);
        let best_configs = g.evolve();

        if best_configs.1 > optimal_params.get(&file_range).unwrap().fitness {
            optimal_params.insert(file_range, ParamsAndFitness { params: best_configs.0, fitness: best_configs.1 });
        }
    }

    // After the loop, write the contents of optimal_params to a file
    let mut file = File::create("optimal_params.txt")?;
    for (range, params_fitness) in &optimal_params {
        writeln!(
            file,
            "{},{},{},{}",
            range.min_size, range.max_size,
            params_fitness.params.threads, params_fitness.params.buffer_size,
        )?;
    }

    Ok(())
}
