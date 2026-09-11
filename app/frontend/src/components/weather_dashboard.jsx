import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { ThermometerSun, Wind, Droplets, Eye, ArrowUp, Scale } from 'lucide-react';

const WeatherDashboard = ({ weatherData }) => {
  if (!weatherData) {
    return <div className="p-4">Loading weather data...</div>;
  }

  const WeatherIcon = ({ condition }) => {
    const iconMap = {
      'clear sky': '☀️',
      'few clouds': '🌤️',
      'scattered clouds': '⛅',
      'broken clouds': '☁️',
      'shower rain': '🌧️',
      'rain': '🌧️',
      'thunderstorm': '⛈️',
      'snow': '🌨️',
      'mist': '🌫️',
      'heavy intensity rain': '⛈️',
      'light rain': '🌦️',
      'overcast clouds': '☁️'
    };
    return <span className="text-2xl">{iconMap[condition.toLowerCase()] || '🌡️'}</span>;
  };

  const formatDateTime = (dateStr) => {
    const date = new Date(dateStr);
    return date.toLocaleString();
  };

  return (
    <div className="space-y-4 p-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <WeatherIcon condition={weatherData.current_conditions.weather} />
            Current Weather Conditions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="flex items-center gap-2">
              <ThermometerSun className="text-blue-500" />
              <div>
                <div className="text-sm text-gray-500">Temperature</div>
                <div className="font-semibold">{weatherData.current_conditions.temperature}°C</div>
                <div className="text-xs text-gray-400">Feels like: {weatherData.current_conditions.feels_like}°C</div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Wind className="text-blue-500" />
              <div>
                <div className="text-sm text-gray-500">Wind</div>
                <div className="font-semibold">{weatherData.current_conditions.wind_speed} m/s</div>
                <div className="text-xs text-gray-400">Direction: {weatherData.current_conditions.wind_direction}°</div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Droplets className="text-blue-500" />
              <div>
                <div className="text-sm text-gray-500">Humidity</div>
                <div className="font-semibold">{weatherData.current_conditions.humidity}%</div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Scale className="text-blue-500" />
              <div>
                <div className="text-sm text-gray-500">Pressure</div>
                <div className="font-semibold">{weatherData.current_conditions.pressure} hPa</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {weatherData.forecast && weatherData.forecast.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>5-Day Forecast</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {weatherData.forecast.map((day, index) => (
                <div key={index} className="flex items-center gap-4 p-2 hover:bg-gray-50 rounded">
                  <WeatherIcon condition={day.weather} />
                  <div className="flex-1">
                    <div className="font-semibold">{formatDateTime(day.datetime)}</div>
                    <div className="text-sm text-gray-500">{day.weather}</div>
                  </div>
                  <div className="text-right">
                    <div className="font-semibold">{day.temperature}°C</div>
                    <div className="text-sm text-gray-500">Humidity: {day.humidity}%</div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {weatherData.seismic_activity && weatherData.seismic_activity.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Recent Seismic Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {weatherData.seismic_activity.map((event, index) => (
                <Alert key={index} className="bg-yellow-50">
                  <AlertTitle className="text-yellow-800">
                    Magnitude {event.magnitude} Earthquake
                  </AlertTitle>
                  <AlertDescription>
                    <div>Location: {event.location}</div>
                    <div>Time: {formatDateTime(event.time)}</div>
                  </AlertDescription>
                </Alert>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default WeatherDashboard;