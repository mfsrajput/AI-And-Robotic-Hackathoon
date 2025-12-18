import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '5ff'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '5ba'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'a2b'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'c3c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '156'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '88c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '000'),
    exact: true
  },
  {
    path: '/',
    component: ComponentCreator('/', '2bc'),
    exact: true
  },
  {
    path: '/',
    component: ComponentCreator('/', 'a05'),
    routes: [
      {
        path: '/',
        component: ComponentCreator('/', 'baa'),
        routes: [
          {
            path: '/',
            component: ComponentCreator('/', '73b'),
            routes: [
              {
                path: '/category/module-1-the-robotic-nervous-system-ros-2',
                component: ComponentCreator('/category/module-1-the-robotic-nervous-system-ros-2', 'a2a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/category/module-2-the-digital-twin-gazebo--unity',
                component: ComponentCreator('/category/module-2-the-digital-twin-gazebo--unity', 'b86'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/category/module-3-the-ai-robot-brain-nvidia-isaac',
                component: ComponentCreator('/category/module-3-the-ai-robot-brain-nvidia-isaac', '2a8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/category/module-4-vision-language-action-vla--capstone',
                component: ComponentCreator('/category/module-4-vision-language-action-vla--capstone', '145'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Chatbot',
                component: ComponentCreator('/Chatbot', 'ae9'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Constitution',
                component: ComponentCreator('/Constitution', '0e2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/contributors',
                component: ComponentCreator('/contributors', '101'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/deployment',
                component: ComponentCreator('/deployment', 'a8a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/digital-twin/creating-complete-digital-twins',
                component: ComponentCreator('/digital-twin/creating-complete-digital-twins', 'd09'),
                exact: true,
                sidebar: "moduleSidebar"
              },
              {
                path: '/digital-twin/gazebo-physics-and-world-building',
                component: ComponentCreator('/digital-twin/gazebo-physics-and-world-building', '508'),
                exact: true,
                sidebar: "moduleSidebar"
              },
              {
                path: '/digital-twin/simulating-sensors-lidar-imu-depth',
                component: ComponentCreator('/digital-twin/simulating-sensors-lidar-imu-depth', 'c48'),
                exact: true,
                sidebar: "moduleSidebar"
              },
              {
                path: '/digital-twin/unity-for-high-fidelity-hri',
                component: ComponentCreator('/digital-twin/unity-for-high-fidelity-hri', 'a39'),
                exact: true,
                sidebar: "moduleSidebar"
              },
              {
                path: '/getting-started',
                component: ComponentCreator('/getting-started', '23e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro',
                component: ComponentCreator('/intro', '9fa'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/local-verification-checklist',
                component: ComponentCreator('/local-verification-checklist', 'a0e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/nvidia-isaac/isaac-ros-vslam-perception',
                component: ComponentCreator('/nvidia-isaac/isaac-ros-vslam-perception', 'cd8'),
                exact: true,
                sidebar: "moduleSidebar"
              },
              {
                path: '/nvidia-isaac/isaac-sim-synthetic-data',
                component: ComponentCreator('/nvidia-isaac/isaac-sim-synthetic-data', '517'),
                exact: true,
                sidebar: "moduleSidebar"
              },
              {
                path: '/nvidia-isaac/nav2-bipedal-locomotion',
                component: ComponentCreator('/nvidia-isaac/nav2-bipedal-locomotion', '8a9'),
                exact: true,
                sidebar: "moduleSidebar"
              },
              {
                path: '/nvidia-isaac/sim-to-real-transfer',
                component: ComponentCreator('/nvidia-isaac/sim-to-real-transfer', 'ecf'),
                exact: true,
                sidebar: "moduleSidebar"
              },
              {
                path: '/preface',
                component: ComponentCreator('/preface', '85b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/RAG-Chatbot-Constitution',
                component: ComponentCreator('/RAG-Chatbot-Constitution', '2bd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ros2/nodes-topics-services-actions',
                component: ComponentCreator('/ros2/nodes-topics-services-actions', 'b87'),
                exact: true,
                sidebar: "moduleSidebar"
              },
              {
                path: '/ros2/python-rclpy-bridge',
                component: ComponentCreator('/ros2/python-rclpy-bridge', 'a5a'),
                exact: true,
                sidebar: "moduleSidebar"
              },
              {
                path: '/ros2/ros2-and-embodied-control',
                component: ComponentCreator('/ros2/ros2-and-embodied-control', '202'),
                exact: true,
                sidebar: "moduleSidebar"
              },
              {
                path: '/ros2/urdf-xacro-for-humanoids',
                component: ComponentCreator('/ros2/urdf-xacro-for-humanoids', 'd41'),
                exact: true,
                sidebar: "moduleSidebar"
              },
              {
                path: '/troubleshooting',
                component: ComponentCreator('/troubleshooting', '810'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/vla/capstone-autonomous-humanoid',
                component: ComponentCreator('/vla/capstone-autonomous-humanoid', '7e1'),
                exact: true,
                sidebar: "moduleSidebar"
              },
              {
                path: '/vla/llm-task-and-motion-planning',
                component: ComponentCreator('/vla/llm-task-and-motion-planning', '667'),
                exact: true,
                sidebar: "moduleSidebar"
              },
              {
                path: '/vla/multi-modal-integration',
                component: ComponentCreator('/vla/multi-modal-integration', '9b7'),
                exact: true,
                sidebar: "moduleSidebar"
              },
              {
                path: '/vla/voice-to-action-with-whisper',
                component: ComponentCreator('/vla/voice-to-action-with-whisper', '875'),
                exact: true,
                sidebar: "moduleSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
