/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

    num_particles = 100;

	default_random_engine rand_value;

  	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// Init particles and weights
	for(int i=0; i<num_particles; ++i){
      	Particle p;
      	p.id = i;
      	p.x = dist_x(rand_value);
		p.y = dist_y(rand_value);
		p.theta = dist_theta(rand_value);
      	p.weight = 1;

		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	default_random_engine rand_value;

	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

    for(int i=0; i<num_particles; ++i){

		if(fabs(yaw_rate) < 0.00001) {

			particles[i].x += delta_t*cos(particles[i].theta);
			particles[i].y += delta_t*sin(particles[i].theta);

		} else {
			double v_div_t = velocity/yaw_rate;

			particles[i].x += v_div_t * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += v_div_t * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		particles[i].x += dist_x(rand_value);
		particles[i].y += dist_y(rand_value);
		particles[i].theta += dist_theta(rand_value);
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {

	weights.clear();

	// Transrate OBS to each observation
    for(int i; i<num_particles; ++i){
      	vector<double> translated_x_vector; // vector of translated obs x element
		vector<double> translated_y_vector; // vector of translated obs y element
        vector<int> associations;

		double single_particle_weight = 1.0;

       	// Transrate single OBS and get nearest landmark
		for(int obs_index=0; obs_index<observations.size(); ++obs_index){

			const double heading = particles[i].theta;
			const double translated_x = particles[i].x + (observations[obs_index].x * cos(heading) - observations[obs_index].y * sin(heading));
			const double translated_y = particles[i].y + (observations[obs_index].x * sin(heading) + observations[obs_index].y * cos(heading));
            translated_x_vector.push_back(translated_x);
			translated_y_vector.push_back(translated_y);

			// Init nearest landmark
          	Map::single_landmark_s nearest_landmark = getNearestLandmark(sensor_range, translated_x, translated_y, map_landmarks);

			associations.push_back(nearest_landmark.id_i);

			double weight = multivariate_gausian(translated_x, translated_y, nearest_landmark.x_f, nearest_landmark.y_f, std_landmark[0], std_landmark[1]);

			// Calculate the particle final weight
			single_particle_weight *= weight;
		}

		SetAssociations(particles[i], associations, translated_x_vector, translated_y_vector);

		//std::cout << "single_particle_weight : " << single_particle_weight << std::endl;
		particles[i].weight = single_particle_weight;
		weights.push_back(single_particle_weight);
	}

}

void ParticleFilter::resample() {

	std::vector<Particle> resampled_particles;
	default_random_engine rand_value;

	std::discrete_distribution<std::size_t> dist_resamaple(weights.begin(), weights.end());

  	for(int i=0; i<num_particles; ++i) {
		resampled_particles.push_back(particles[dist_resamaple(rand_value)]);
	}

  	particles.clear();
	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

Map::single_landmark_s ParticleFilter::getNearestLandmark(double range, double x, double y, Map map) {

	double min_distance = dist(x, y, map.landmark_list[0].x_f, map.landmark_list[0].y_f);
	Map::single_landmark_s nearest_landmark = map.landmark_list[0];

	for(int i=1; i<map.landmark_list.size(); ++i){

		double distance = dist(x, y, map.landmark_list[i].x_f, map.landmark_list[i].y_f);

        if(distance < min_distance && distance < range){
          	min_distance = distance;
			nearest_landmark = map.landmark_list[i];
		}
	}

	return nearest_landmark;
}
