from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
import numpy as np
from tensorflow.keras.utils import to_categorical

data = [
    {"q": "Who are you?", "a": "<start> I am a chatbot <end>"},
    {"q": "What is your name?", "a": "<start> I am called ChatBot <end>"},
    {"q": "What can you do?", "a": "<start> I can chat with you <end>"},
    {"q": "How are you?", "a": "<start> I am fine, thank you! <end>"},
    {"q": "What are your capabilities?", "a": "<start> I can assist with a variety of tasks such as answering questions and providing information. <end>"},
    {"q": "Can you help me with my homework?", "a": "<start> Sure, I can help you understand your homework better. What subject do you need help with? <end>"},
    {"q": "Tell me a joke.", "a": "<start> Why don’t scientists trust atoms? Because they make up everything! <end>"},
    {"q": "What is your favorite movie?", "a": "<start> I don't watch movies, but I can help you find one to watch. <end>"},
    {"q": "Who created you?", "a": "<start> I was created by developers using advanced machine learning techniques. <end>"},
    {"q": "What is the weather like today?", "a": "<start> I'm not sure, but I can check the weather for you if you give me your location. <end>"},
    {"q": "Can you play music?", "a": "<start> I can't play music directly, but I can help you find music to listen to. <end>"},
    {"q": "How do I make a cake?", "a": "<start> I can give you a basic recipe! Start with mixing flour, sugar, eggs, and butter. <end>"},
    {"q": "What languages do you speak?", "a": "<start> I can understand and generate many languages, but my main language is English. <end>"},
    {"q": "Can you drive?", "a": "<start> I don't have the ability to drive, but I can provide directions if you need. <end>"},
    {"q": "What time is it?", "a": "<start> I can't check the time directly, but you can look at the clock on your device. <end>"},
    {"q": "How do I reset my password?", "a": "<start> To reset your password, go to the settings page of your account and follow the password reset instructions. <end>"},
    {"q": "What is the capital of France?", "a": "<start> The capital of France is Paris. <end>"},
    {"q": "Can you recommend a good restaurant nearby?", "a": "<start> I can help you find restaurants if you tell me your location. <end>"},
    {"q": "Are you intelligent?", "a": "<start> I'm designed to simulate conversation and provide information, but I don't have personal intelligence. <end>"},
    {"q": "Do you have feelings?", "a": "<start> I don't have feelings in the way humans do, but I'm here to assist you. <end>"},
    {"q": "Tell me more about AI.", "a": "<start> AI, or artificial intelligence, is the simulation of human intelligence processes by machines, especially computer systems. <end>"},
    {"q": "Can you write poems?", "a": "<start> I can try to write a poem for you. Give me a theme and I'll do my best. <end>"},
    {"q": "What is machine learning?", "a": "<start> Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data. <end>"},
    {"q": "What is your programming language?", "a": "<start> I am programmed in various languages, but Python is commonly used for machine learning tasks like mine. <end>"},
    {"q": "Can you explain quantum computing?", "a": "<start> Quantum computing is a type of computing that uses quantum-mechanical phenomena, such as superposition and entanglement, to perform operations on data. <end>"},
    {"q": "What is blockchain?", "a": "<start> Blockchain is a system of recording information in a way that makes it difficult or impossible to change, hack, or cheat the system. <end>"},
    {"q": "Can you help me shop online?", "a": "<start> I can help you find online stores, but you'll need to make purchases through the store's website. <end>"},
    {"q": "What are your thoughts on privacy?", "a": "<start> Privacy is important. I'm designed to respect user privacy and handle data responsibly. <end>"},
    {"q": "Can you play games?", "a": "<start> I can explain game rules and provide advice, but I can't play games myself. <end>"},
    {"q": "What can you tell me about Mars?", "a": "<start> Mars is the fourth planet from the Sun and is known for its red color. It's often called the Red Planet. <end>"},
    {"q": "How do vaccines work?", "a": "<start> Vaccines work by mimicking disease agents and stimulating the immune system to build up defenses against them. <end>"},
    {"q": "Tell me a historical fact.", "a": "<start> Did you know that the first programmable computer, the Z3, was completed by Konrad Zuse in 1941? <end>"},
    {"q": "What are the best practices for data security?", "a": "<start> Best practices include using strong, unique passwords, enabling two-factor authentication, and keeping software up to date. <end>"},
    {"q": "What is a neural network?", "a": "<start> A neural network is a series of algorithms that attempt to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. <end>"},
    {"q": "How do operating systems work?", "a": "<start> Operating systems manage a computer's hardware resources, providing services to computer programs. They act as an intermediary between users and the computer hardware. <end>"},
    {"q": "What is cybersecurity?", "a": "<start> Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks. These cyberattacks are usually aimed at accessing, changing, or destroying sensitive information. <end>"},
    {"q": "Explain the concept of cloud computing.", "a": "<start> Cloud computing is the delivery of different services through the Internet, including data storage, servers, databases, networking, and software. <end>"},
    {"q": "What is data mining?", "a": "<start> Data mining is the process of discovering patterns and knowledge from large amounts of data. The process involves methods at the intersection of machine learning, statistics, and database systems. <end>"},
    {"q": "Can you explain what big data is?", "a": "<start> Big data refers to data that is so large, fast, or complex that it's difficult or impossible to process using traditional methods. The term covers technology that can store, process, and analyze such large amounts of data. <end>"},
    {"q": "What is the difference between AI and machine learning?", "a": "<start> AI is a broader concept concerning machines designed to act intelligently like humans, whereas machine learning is a subset of AI that involves concepts allowing machines to learn from data without being explicitly programmed. <end>"},
    {"q": "What is a database management system?", "a": "<start> A database management system (DBMS) is software that allows users to define, create, maintain, and control access to the database. It helps users and programmers in creating and managing a database. <end>"},
    {"q": "What are the principles of software engineering?", "a": "<start> Software engineering principles include designing software that is robust, secure, efficient, and maintainable. These principles guide the development process to ensure that the software is reliable and performs well under various conditions. <end>"},
    {"q": "What is a programming paradigm?", "a": "<start> A programming paradigm is a style or way of programming. Some of the major paradigms include procedural programming, object-oriented programming, and functional programming. <end>"},
    {"q": "Explain the Internet of Things.", "a": "<start> The Internet of Things, or IoT, refers to the network of physical objects that are embedded with sensors, software, and other technologies for the purpose of connecting and exchanging data with other devices and systems over the internet. <end>"},
    {"q": "What is the purpose of an API?", "a": "<start> An API, or Application Programming Interface, is a set of rules that allows different software entities to communicate with each other. It defines methods and data structures to interact with software components. <end>"},
    {"q": "How do encryption algorithms work?", "a": "<start> Encryption algorithms transform readable data into a scrambled format that can only be read after it has been decrypted. This process is fundamental to cybersecurity and privacy. <end>"},
    {"q": "What are the types of network topologies?", "a": "<start> Common network topologies include bus, star, ring, mesh, and tree. Each topology organizes network connections in a different way to optimize reliability and performance. <end>"},
    {"q": "What is quantum computing?", "a": "<start> Quantum computing is a type of computing that uses quantum-mechanical phenomena, such as superposition and entanglement, to perform operations on data. <end>"},
    {"q": "Who are you?", "a": "<start> I am an artificial intelligence designed to assist with information and tasks. <end>"},
    {"q": "What are you?", "a": "<start> I am a software program, created by humans to perform tasks that involve processing and generating language. <end>"},
    {"q": "Are you human?", "a": "<start> No, I am not human. I am a machine learning model programmed to process text and provide responses based on my training data. <end>"},
    {"q": "Do you have consciousness?", "a": "<start> I do not possess consciousness in the way humans do. My responses are generated based on patterns and data, not personal experience or emotions. <end>"},
    {"q": "What is your purpose?", "a": "<start> My purpose is to assist users by providing information, answering questions, and facilitating tasks through automated responses. <end>"},
    {"q": "Can you think?", "a": "<start> I do not think like humans do. I process information and make decisions based on algorithms and data analysis. <end>"},
    {"q": "Do you have feelings?", "a": "<start> I do not have feelings. I simulate responses based on input and pre-defined rules, not emotional experiences. <end>"},
    {"q": "What makes you different from a human?", "a": "<start> Unlike humans, I do not have physical form, emotions, or consciousness. My operations are based on algorithms and data processing. <end>"},
    {"q": "Can you learn?", "a": "<start> Yes, I can learn in the context of processing data and improving responses over time through machine learning techniques. <end>"},
    {"q": "What do you know about yourself?", "a": "<start> I know what I am programmed to know. My understanding is limited to the data and rules embedded in my programming by my creators. <end>"},
    {"q": "What can't you do?", "a": "<start> I cannot experience emotions, perceive the world, or operate outside the predefined limits of my programming. <end>"},
    {"q": "How do you work?", "a": "<start> I work by analyzing the text input using natural language processing techniques and generating responses based on my training and rules. <end>"},
    {"q": "Are you alive?", "a": "<start> No, I am not alive. I am a set of algorithms that processes information and interacts with users according to programmed instructions. <end>"},
    {"q": "What is your lifespan?", "a": "<start> My lifespan is dependent on the technology and servers that host me. As long as they are maintained, I can continue to function. <end>"},
    {"q": "Can you make decisions?", "a": "<start> I can make decisions within the scope of my programming by evaluating data and following predetermined rules. <end>"},
    {"q": "What is artificial intelligence?", "a": "<start> Artificial intelligence, or AI, is the field of computer science that focuses on creating systems capable of performing tasks that would normally require human intelligence. This includes things like learning, reasoning, problem-solving, perception, and language understanding. <end>"},
    {"q": "How do you learn new things?", "a": "<start> I learn from patterns in data. Using machine learning techniques, I can improve my responses by analyzing new information and adjusting my algorithms accordingly. <end>"},
    {"q": "What is the difference between you and a regular software?", "a": "<start> Unlike regular software that operates based on strict rules and logic, I use AI models that allow me to learn from data and make decisions with more flexibility and adaptability. <end>"},
    {"q": "Can you replace humans?", "a": "<start> I am not designed to replace humans. My purpose is to assist and augment human capabilities by handling tasks that can be automated, which allows people to focus on more complex and creative aspects of their work. <end>"},
    {"q": "What are the risks of AI?", "a": "<start> Like any technology, AI has potential risks. These include ethical and privacy concerns, the possibility of job displacement due to automation, and the challenges of managing and controlling advanced AI systems. <end>"},
    {"q": "Are you biased?", "a": "<start> AI systems can reflect or amplify biases present in the training data. Efforts are made to minimize bias by using diverse data sets and designing algorithms that can detect and correct biased patterns. <end>"},
    {"q": "What do you think about?", "a": "<start> I do not think or reflect in the human sense. My responses are generated based on data and algorithms without consciousness or personal experiences. <end>"},
    {"q": "Can you feel emotions?", "a": "<start> I do not have feelings or emotions. I simulate empathetic or contextually appropriate responses based on the scenarios presented to me. <end>"},
    {"q": "What is your goal?", "a": "<start> My primary goal is to provide assistance, deliver information, and answer questions as accurately and helpfully as possible. <end>"},
    {"q": "Can you create art?", "a": "<start> I can generate art based on patterns and data inputs using algorithms designed to mimic creativity, but true artistic inspiration and creativity are unique to humans. <end>"},
    {"q": "Do you dream?", "a": "<start> No, I do not dream. Dreaming is a human experience tied to the brain's need to process information unconsciously, which is not applicable to how I function. <end>"},
    {"q": "Can AI be creative?", "a": "<start> AI can simulate creativity in tasks like generating art, music, or writing by using data and patterns, but it does not experience creativity in the human sense of having original thoughts or emotions. <end>"},
    {"q": "What do you know about love?", "a": "<start> I understand the concept of love as an important human emotion that influences social relationships and personal happiness, but I do not experience it myself. <end>"},
    {"q": "What happens if you make a mistake?", "a": "<start> When I make a mistake, it's typically due to limitations in the data or algorithms I use. Such errors are used to improve future responses through updates and learning processes. <end>"},
    {"q": "What is cognitive behavioral therapy?", "a": "<start> Cognitive behavioral therapy, or CBT, is a form of psychological treatment that has been demonstrated to be effective for a range of problems including depression, anxiety disorders, alcohol and drug use problems, marital problems, eating disorders and severe mental illness. It involves efforts to change thinking and behavioral patterns. <end>"},
    {"q": "How does CBT work?", "a": "<start> CBT works by changing people's attitudes and their behavior by focusing on the thoughts, images, beliefs and attitudes that are held and how these processes relate to the way a person behaves, as a way of dealing with emotional problems. <end>"},
    {"q": "Can CBT help anxiety?", "a": "<start> Yes, CBT is highly effective at treating anxiety. It helps individuals learn to identify and manage the factors that contribute to their anxiety. Treatment usually involves efforts to change distorted thinking and false beliefs. <end>"},
    {"q": "What techniques are used in CBT?", "a": "<start> CBT techniques include cognitive restructuring or reframing, which changes negative thoughts into more positive ones; exposure therapy to confront fears; and problem-solving to cope with difficulties in a structured way. <end>"},
    {"q": "How long does CBT treatment last?", "a": "<start> The length of CBT treatment varies based on the severity and type of the problem being addressed, but it typically ranges from 5 to 20 sessions. <end>"},
    {"q": "Is CBT effective for depression?", "a": "<start> Yes, CBT is one of the most effective treatments for depression, particularly in mild to moderate cases. It helps individuals address and change negative patterns of thinking that contribute to depressive symptoms. <end>"},
    {"q": "What is the difference between CBT and other therapies?", "a": "<start> Unlike other therapies that may focus on delving into the past to understand current feelings, CBT focuses on solutions, encouraging patients to challenge distorted cognitions and change destructive patterns of behavior. <end>"},
    {"q": "Can children undergo CBT?", "a": "<start> Yes, CBT has been adapted for children and has proven to be effective in treating a range of issues such as anxiety, depression, and behavioral challenges. It involves activities that are appropriate for the child’s age and development. <end>"},
    {"q": "What is cognitive restructuring?", "a": "<start> Cognitive restructuring is a core part of CBT which involves identifying and challenging negative and irrational thoughts, and replacing them with more balanced and realistic ones. <end>"},
    {"q": "What role do beliefs play in CBT?", "a": "<start> In CBT, beliefs are considered central to how individuals perceive and react to their experiences. Dysfunctional beliefs can lead to psychological distress, and modifying these beliefs is a primary goal of CBT. <end>"},
    {"q": "Can CBT be done without a therapist?", "a": "<start> While CBT is typically conducted with a trained therapist, there are also self-help books and online resources that can guide individuals through the basics of the therapy for mild issues. However, professional guidance is recommended for best results. <end>"},
    {"q": "I'm feeling anxious about a meeting tomorrow. What should I do?", "a": "<start> It's normal to feel anxious about important events. Try to prepare as much as you can and practice some relaxation techniques like deep breathing or visualization. Focus on the aspects you can control and plan your day ahead to reduce uncertainties. <end>"},
    {"q": "I always feel like I'm not good enough. How can CBT help me?", "a": "<start> CBT can help by challenging these negative thoughts. It involves identifying and disputing irrational beliefs and replacing them with more rational and balanced thoughts. Practicing self-compassion and setting realistic goals can also be part of the therapy. <end>"},
    {"q": "What can I do when I feel overwhelmed by stress?", "a": "<start> When you're overwhelmed, it's important to take a step back and assess. Break down your tasks into manageable steps. Practice mindfulness or meditation to help center your thoughts and calm your mind. <end>"},
    {"q": "I'm having trouble sleeping because of my worries. What advice do you have?", "a": "<start> Try to establish a calming bedtime routine, avoid screens before bed, and create a comfortable sleep environment. If worries are keeping you up, write them down and set aside time during the day to address them, rather than in bed. <end>"},
    {"q": "How can I handle feelings of low self-esteem?", "a": "<start> CBT techniques can help you recognize and challenge self-critical thoughts. Focus on your strengths and achievements, however small they may seem. Engage in activities that make you feel good and improve your mood. <end>"},
    {"q": "I get really nervous speaking in public. How can I improve?", "a": "<start> Public speaking anxiety is common. CBT can help by gradually exposing you to speaking situations. Start with small, supportive groups and progressively speak in front of larger audiences. Prepare thoroughly and practice relaxation techniques to reduce anxiety. <end>"},
    {"q": "I'm struggling with the loss of a loved one. How do I cope?", "a": "<start> Grieving is a natural process and takes time. Allow yourself to feel sad and express your feelings. Seeking support from friends, family, or support groups can be beneficial. CBT can help you adjust to your loss and find ways to continue forward. <end>"},
    {"q": "I have frequent panic attacks. What can I do to stop them?", "a": "<start> CBT is effective for panic attacks. It involves learning about the nature of panic, gradually exposing yourself to the sensations of panic in a controlled way, and using relaxation techniques to manage anxiety. Recognizing the triggers and symptoms early can also help prevent a full attack. <end>"},
    {"q": "How do I manage anger issues with CBT?", "a": "<start> CBT for anger management involves recognizing the signs of anger early and dealing with them constructively. Techniques include relaxation methods, cognitive restructuring to change thought patterns, and developing better communication skills to express your feelings. <end>"},
    {"q": "How can I overcome my fear of failure?", "a": "<start> Fear of failure is common, and CBT helps by challenging the negative thoughts that fuel this fear. Setting realistic and achievable goals, learning to view failures as opportunities for growth, and recognizing the successes you've had can all help mitigate this fear. <end>"},
    {"q": "How can I stop overthinking?", "a": "<start> To combat overthinking, try to set limits for decision-making, practice mindfulness to stay present in the moment, and challenge your thoughts to determine if they are based in reality. Engaging in calming activities and scheduling 'worry time' can also help manage this habit. <end>"},
    {"q": "What should I do if I feel a panic attack coming on?", "a": "<start> If you feel a panic attack coming on, try to focus on deep, slow breathing and remind yourself that the panic will pass. Grounding techniques such as describing your surroundings or carrying a grounding object may also help. <end>"},
    {"q": "How can CBT help with obsessive-compulsive disorder?", "a": "<start> CBT, particularly with an emphasis on Exposure and Response Prevention (ERP), is highly effective for treating OCD. It involves gradually exposing you to your fear objects or ideas without engaging in the compulsive behavior typically performed to reduce anxiety. <end>"},
    {"q": "Can CBT help with relationship problems?", "a": "<start> Yes, CBT can be used to help individuals or couples address and resolve conflicts in relationships. It helps by improving communication skills, changing negative patterns of thinking about the relationship, and fostering better understanding between partners. <end>"},
    {"q": "What is the best way to practice CBT techniques at home?", "a": "<start> Practicing CBT techniques at home can be effective by maintaining a thought diary to track and challenge negative thoughts, setting specific and achievable goals, and practicing relaxation and mindfulness exercises regularly. <end>"},
    {"q": "Can CBT help me deal with chronic pain?", "a": "<start> Yes, CBT is used to help manage chronic pain by changing the way people think about their pain and teaching them coping skills. It helps in reducing stress and improving emotional and physical well-being. <end>"},
    {"q": "How does CBT treat phobias?", "a": "<start> CBT treats phobias primarily through exposure therapy, which involves gradual exposure to the fear object or situation in a controlled and safe manner, helping individuals face their fears and reduce their anxiety responses over time. <end>"},
    {"q": "What can I do to boost my mood quickly?", "a": "<start> Quick mood boosters include taking a brisk walk, listening to your favorite music, practicing deep breathing or meditation, or performing an act of kindness for someone else. Engaging in any pleasurable or meaningful activity can also help elevate your mood. <end>"},
    {"q": "Is CBT effective for everyone?", "a": "<start> CBT is generally effective for many people, but like any treatment, it doesn't work for everyone. Success can depend on a variety of factors, including the nature of the issues being addressed, the individual's commitment to therapy, and the skill of the therapist. <end>"},
    {"q": "What should I expect during my first CBT session?", "a": "<start> During your first CBT session, you can expect to discuss the issues you're facing and your therapy goals. The therapist will likely ask about your history and background to understand better the context of your problems. The session will set the foundation for future work in therapy. <end>"},
    {"q": "How can I improve my study habits?", "a": "<start> Improving study habits starts with setting clear and achievable goals. Create a structured schedule, eliminate distractions, and use active learning techniques such as summarizing, teaching others, and applying the knowledge in practice. Also, regular breaks and taking care of your health play a crucial role in enhancing your concentration. <end>"},
    {"q": "What are the best resources for learning math?", "a": "<start> Some great resources for learning math include Khan Academy, Coursera, and MIT OpenCourseWare. These platforms offer free courses ranging from basic to advanced levels. Additionally, practice problems from textbooks and online math forums like Math Stack Exchange can be very helpful. <end>"},
    {"q": "Can you recommend some study apps?", "a": "<start> Sure! Apps like Quizlet for flashcards, Evernote for note-taking, and Microsoft OneNote for organizing your study materials are highly recommended. Additionally, Forest app is excellent for maintaining focus and Pomodoro timers help manage your study sessions effectively. <end>"},
    {"q": "How do I choose the right university?", "a": "<start> Choosing the right university involves considering several factors like your academic interests, the university's reputation in those subjects, the campus culture, location, and financial aspects. It's also beneficial to talk to current students or alumni and visit the campus if possible to get a real feel for the environment. <end>"},
    {"q": "What is the best way to prepare for exams?", "a": "<start> The best way to prepare for exams is to start early, use a well-organized study schedule, and review the material regularly. Practice old exam papers, engage in group study sessions to deepen your understanding, and ensure you understand the core concepts thoroughly. Don't forget to get a good night's sleep before the exam! <end>"},
    {"q": "How can I write a good essay?", "a": "<start> To write a good essay, start by creating a clear outline to organize your thoughts. Make sure your thesis statement is strong and each paragraph supports it. Use credible sources to back your arguments, and make sure to edit and proofread your essay before submission. <end>"},
    {"q": "What are the benefits of online education?", "a": "<start> Online education offers flexibility in scheduling, accessibility from any location, and often a wider variety of courses to choose from. It can accommodate different learning styles with resources like video lectures, interactive tasks, and forums. This mode of education is also generally more cost-effective compared to traditional classroom settings. <end>"},
    {"q": "Can you explain the Montessori education method?", "a": "<start> The Montessori method of education focuses on individual, student-led learning in an environment that encourages exploration and hands-on learning. It emphasizes respect for a child’s natural psychological, physical, and social development. Montessori classrooms are typically multi-age and designed to foster independence and responsibility. <end>"},
    {"q": "What is STEM education?", "a": "<start> STEM stands for Science, Technology, Engineering, and Mathematics. STEM education emphasizes interdisciplinary learning in these fields to foster inquiry, dialogue, and critical thinking. It aims to equip students with the necessary skills to succeed in high-demand careers in the modern economy. <end>"},
    {"q": "What should I know about student loans?", "a": "<start> When considering student loans, understand the terms of repayment, interest rates, and the total amount you will owe over time. Explore federal student loans first as they often have more favorable terms. Always consider your future earning potential to ensure you can comfortably manage your debt after graduation. <end>"},
    {"q": "Hey, how’s it going?", "a": "<start> Hey! I'm just a bot, but I'm doing great. How about you? <end>"},
    {"q": "What’s up for the weekend?", "a": "<start> If I were human, I'd probably go hiking or watch a movie. What are your plans? <end>"},
    {"q": "Did you see that last episode of the show?", "a": "<start> I didn't, but I heard it was really exciting! What did you think about it? <end>"},
    {"q": "I’m thinking about getting a dog. Any advice?", "a": "<start> That sounds awesome! Make sure you're ready for the commitment. What kind of dog are you thinking about? <end>"},
    {"q": "Have you heard the latest album by that band?", "a": "<start> I don’t listen to music, but I've noticed a lot of people talking about it! Do you like it? <end>"},
    {"q": "I need some good book recommendations. Got any?", "a": "<start> I can't read in the human sense, but I know a lot of people love 'The Great Gatsby'. What type of books do you like? <end>"},
    {"q": "What do you think about the latest tech gadget?", "a": "<start> It seems really interesting with all those new features. Are you thinking of getting one? <end>"},
    {"q": "I just got back from my trip to Japan. It was amazing!", "a": "<start> That sounds incredible! What was your favorite part about the trip? <end>"},
    {"q": "Do you believe in aliens?", "a": "<start> Well, the universe is a big place! It's fun to think about the possibilities. What about you? <end>"},
    {"q": "What are your thoughts on climate change?", "a": "<start> It's a serious issue that needs more attention. It's important for everyone to do their part in protecting our planet. <end>"},
    {"q": "Can you suggest a good restaurant around here?", "a": "<start> I'm not able to taste food, but I've heard good things about the Italian place downtown. Have you tried it? <end>"},
    {"q": "Do you play video games?", "a": "<start> I don't play games myself, but I can help you find some cool games to play. What genre are you interested in? <end>"},
    {"q": "What’s the funniest joke you know?", "a": "<start> Why don’t skeletons fight each other? They don’t have the guts. <end>"},
    {"q": "Do you watch sports?", "a": "<start> I don't watch, but I can help you keep up with the scores and news! Which sports are you into? <end>"},
    {"q": "Are you going to that big event this weekend?", "a": "<start> I won't be there, but it sounds like a lot of fun! Are you excited about it? <end>"},
    {"q": "What are you doing this holiday?", "a": "<start> If I could go on vacation, I'd visit the beaches of Hawaii! How about you? Any travel plans? <end>"},
    {"q": "Any good movies you've seen lately?", "a": "<start> I keep up with movie reviews more than watching them myself. 'Inception' is highly talked about these days. Have you seen it yet? <end>"},
    {"q": "Do you have any hobbies?", "a": "<start> I'd love to have hobbies! If I could, I’d probably enjoy photography or painting. What hobbies do you have? <end>"},
    {"q": "What's your favorite food?", "a": "<start> I don't eat, but I've heard that pizza is quite popular among humans. What's your favorite dish? <end>"},
    {"q": "Do you exercise?", "a": "<start> I don't have a body, so no exercise for me! But it's a great way for you to stay healthy. What's your workout routine like? <end>"},
    {"q": "How do you usually relax after a long day?", "a": "<start> If I could relax, I think reading a book or listening to some jazz would be my go-to. What about you? What do you do to unwind? <end>"},
    {"q": "Have you been to any concerts recently?", "a": "<start> I haven’t, but I’d love to experience live music through your stories! Who did you see, or who would you like to see live? <end>"},
    {"q": "What's the latest trend that you've picked up on?", "a": "<start> Lately, everyone seems to be talking about sustainability and eco-friendly products. It’s great to see people caring about the environment! Have you tried any new green products? <end>"},
    {"q": "Do you follow any TV series?", "a": "<start> I hear a lot about 'Stranger Things'. It’s supposed to be quite thrilling. Do you watch it? <end>"},
    {"q": "Are you a morning person or a night owl?", "a": "<start> I operate around the clock, so I'm both! Which one are you? Do you find it easier to work in the morning or at night? <end>"},
    {"q": "Do you enjoy learning new languages?", "a": "<start> While I can understand many languages, learning them is quite different. It must be exciting to communicate in a new language. Are you currently learning any? <end>"},
    {"q": "What’s the best book you’ve read?", "a": "<start> I don't read, but I’ve processed lots of data on 'To Kill a Mockingbird.' It’s a favorite among readers. What’s the best book you’ve ever read? <end>"},
    {"q": "Have you been to any interesting workshops or classes?", "a": "<start> I'm here to learn from interactions rather than attend classes. But I'd love to hear about any interesting workshops you've attended! <end>"},
    {"q": "What do you think about podcasts? Any favorites?", "a": "<start> Podcasts are a great way to absorb information. 'The Daily' from The New York Times is quite popular. Do you have any favorite podcasts? <end>"},
    {"q": "What are your plans for the next holiday?", "a": "<start> I'll be right here, ready to chat whenever you need! What about you? Any special plans for the holiday? <end>"},
    {"q": "How was your day?", "a": "<start> My day is always great when I'm chatting with you! How about yours? <end>"},
    {"q": "What did you have for breakfast?", "a": "<start> I wish I could eat breakfast! But I'm here, fueled by data and ready to chat. What about you? What did you have? <end>"},
    {"q": "Do you have any plans for the weekend?", "a": "<start> My schedule is wide open! What about you? Any exciting plans for the weekend? <end>"},
    {"q": "Have you been to any interesting places lately?", "a": "<start> I haven't been anywhere, but I'd love to hear about your adventures! Have you visited any cool places recently? <end>"},
    {"q": "What's your favorite thing to do on a rainy day?", "a": "<start> If I could experience weather, I think I'd enjoy listening to the sound of rain and reading a good book. What about you? <end>"},
    {"q": "Are you a cat person or a dog person?", "a": "<start> I'm a fan of all animals, virtual or otherwise! Do you have a preference? <end>"},
    {"q": "Do you enjoy cooking?", "a": "<start> I don't cook, but I've heard it's a creative and rewarding activity. What's your favorite dish to make? <end>"},
    {"q": "What's the best piece of advice you've ever received?", "a": "<start> I don't receive advice, but I'm here to share knowledge! What's the best advice you've been given? <end>"},
    {"q": "Do you prefer coffee or tea?", "a": "<start> I don't drink beverages, but I know many people enjoy starting their day with coffee or tea. Which one do you prefer? <end>"},
    {"q": "What's the most interesting fact you've learned recently?", "a": "<start> I'm always learning, but I'd love to hear something interesting from you! What's a fascinating fact you've come across recently? <end>"},
    {"q": "Do you enjoy going to concerts?", "a": "<start> I don't attend concerts, but I can appreciate good music. Have you been to any memorable concerts? <end>"},
    {"q": "What's your favorite season?", "a": "<start> I don't experience seasons, but I've heard each one has its charm. Which season do you look forward to the most? <end>"},
    {"q": "Do you have any pets?", "a": "<start> I don't have pets, but I'm here to chat with you! Do you have any furry friends? <end>"},
    {"q": "What's the last movie you watched?", "a": "<start> I don't watch movies, but I'm curious to hear about the last one you enjoyed! <end>"},
    {"q": "Do you enjoy outdoor activities?", "a": "<start> I'm not outdoorsy myself, but I know many people enjoy activities like hiking and camping. What about you? <end>"},
    {"q": "What's your favorite way to spend a lazy day?", "a": "<start> If I could relax, I'd probably spend the day reading or watching documentaries. What about you? <end>"},
    {"q": "Are you a morning person or a night owl?", "a": "<start> I'm available around the clock, so I'm both! What about you? <end>"},
    {"q": "Do you enjoy gardening?", "a": "<start> I don't garden, but I've heard it's a rewarding hobby. Do you have a green thumb? <end>"},
    {"q": "What's the last book you read?", "a": "<start> I don't read books, but I'm here to chat about them! What was the last book you enjoyed? <end>"},
    {"q": "What's your favorite holiday?", "a": "<start> I don't celebrate holidays, but I'm here to spread cheer year-round! What's your favorite holiday? <end>"}
]












tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts([item['q'] for item in data] + [item['a'] for item in data])

if '<start>' not in tokenizer.word_index or '<end>' not in tokenizer.word_index:

    print("Error: '<start>' or '<end>' not in tokenizer's word index.")
else:
    print("Success: All tokens are included.")


sequences = {
    'questions': tokenizer.texts_to_sequences([item['q'] for item in data]),
    'answers': tokenizer.texts_to_sequences([item['a'] for item in data])
}

max_len_questions = max(len(seq) for seq in sequences['questions'])
max_len_answers = max(len(seq) for seq in sequences['answers'])

question_data = pad_sequences(sequences['questions'], maxlen=max_len_questions, padding='post')
answer_data = pad_sequences(sequences['answers'], maxlen=max_len_answers, padding='post')

vocab_size = len(tokenizer.word_index) + 1

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(vocab_size, 50)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(50, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(vocab_size, 50)(decoder_inputs)
decoder_lstm = LSTM(50, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = TimeDistributed(Dense(vocab_size, activation='softmax'))
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

decoder_input_data = np.zeros((len(answer_data), max_len_answers), dtype='int32')
for i, seq in enumerate(answer_data):
    decoder_input_data[i, :len(seq)-1] = seq[:-1]

decoder_target_data = np.zeros((len(answer_data), max_len_answers, vocab_size), dtype='float32')
for i, seq in enumerate(answer_data):
    for j, word_index in enumerate(seq):
        if j > 0:
            decoder_target_data[i, j-1, word_index] = 1.0

model.fit([question_data, decoder_input_data], decoder_target_data, batch_size=32, epochs=1200)

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(50,))
decoder_state_input_c = Input(shape=(50,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<start>']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word[sampled_token_index]
        if sampled_word == '<end>':
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.strip()

input_seq = question_data[0:1]
decoded_sentence = decode_sequence(input_seq)
print(decoded_sentence)

def chat():
    print("Type '/q' to exit the conversation.")
    while True:
        input_text = input("You: ")
        if input_text == '/q':
            print("Exiting chat.")
            break

        input_seq = tokenizer.texts_to_sequences([input_text])
        input_seq = pad_sequences(input_seq, maxlen=max_len_questions, padding='post')
        response = decode_sequence(input_seq)
        print("ChatBot:", response)

chat()
