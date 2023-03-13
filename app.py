import pickle 
import streamlit as st
import pandas as pd



st.title('The Big 5 Personality Traits Model')

#st.subheader('Understand yourself better!')



def main_page():
    
    st.markdown("## What is Personality?")
    st.write("""        
Personality refers to the unique set of characteristics, traits, behaviors, and patterns
of thought that define an individual and distinguish them from others. It is the way
in which we perceive, interact with, and respond to the world around us.
Personality is believed to be a combination of genetic, environmental, and social
factors, and it can influence every aspect of our lives, from our relationships and
career choices to our emotional well-being and overall happiness.""")

    st.markdown("## Why is it important?")
    st.write("""
Everyone should be aware of their own personality traits because it can help them
understand themselves better, improve their communication and relationships with
others, and make informed decisions about their personal and professional lives.
By understanding your own personality traits, you can identify your strengths and
weaknesses, and work on areas that need improvement. It can also help you
understand how you react to different situations and why you may be more drawn
to certain activities or types of people.
In addition, being aware of your own personality can help you better understand
and interact with others. By recognizing different personality traits in others, you
can adjust your communication style and behavior to better fit their needs and
preferences, leading to more harmonious relationships and effective collaboration.
Overall, understanding your own personality can lead to increased self-awareness,
personal growth, and more fulfilling relationships and experiences in life.""")

    st.markdown("## What is the Big 5 Personality Traits Model?")
    st.write("""

The Big Five personality traits model is a widely recognized framework for
describing and understanding human personality. It is also known as the Five
Factor Model (FFM) and consists of five broad dimensions of personality. These
dimensions are openness to experience, conscientiousness, extraversion,
agreeableness, and neuroticism. Let's explore each of these dimensions in detail:
1. Openness to experience: This dimension refers to a person's openness and
receptivity to new experiences, ideas, and perspectives. People who score
high in openness tend to be curious, imaginative, creative, and open-minded.
They enjoy exploring new ideas and concepts and are often interested in art,
music, and culture. People who score low in openness tend to be more
traditional, practical, and conventional.
2. Conscientiousness: This dimension refers to a person's level of organization,
responsibility, and self-discipline. People who score high in
conscientiousness tend to be reliable, organized, and hard-working. They set
high standards for themselves and take their obligations seriously. People
who score low in conscientiousness tend to be more relaxed, spontaneous,
and carefree.
3. Extraversion: This dimension refers to a person's level of sociability,
assertiveness, and energy. People who score high in extraversion tend to be
outgoing, talkative, and energetic. They enjoy being around others and often
seek out social interactions. People who score low in extraversion tend to be
more reserved, quiet, and introspective.
4. Agreeableness: This dimension refers to a person's level of empathy,
cooperation, and kindness. People who score high in agreeableness tend to
be compassionate, cooperative, and caring. They value harmony and tend to
be supportive of others. People who score low in agreeableness tend to be
more competitive, assertive, and sometimes critical.
5. Neuroticism: This dimension refers to a person's level of emotional stability,
anxiety, and sensitivity. People who score high in neuroticism tend to be
more anxious, worried, and emotionally reactive. They may be more
sensitive to stress and may experience a wider range of emotions. People
who score low in neuroticism tend to be more emotionally stable and
resilient.""")
    st.write("""
The Big Five personality traits model is widely used in psychology research and
has been found to be reliable and valid across different cultures and populations. It
can be useful in many areas, including personal growth, career development, and
relationships. By understanding our own personality traits and those of others, we
can develop greater self-awareness and improve our interactions with others.
The Big Five personality traits model works by measuring an individual's level of
each of the five dimensions of personality: openness to experience,
conscientiousness, extraversion, agreeableness, and neuroticism. There are various
methods used to measure these dimensions, including self-report questionnaires,
observer ratings, and interviews.
    
    """)

def test():
    
    st.markdown("## Get to know yourself better!")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Extraversion", "Neuroticism", "Agreeableness", "Conscientiousness", "Openness to Experience", "Results"])
    tab1.write("The following questions test the Extraversion Trait")
    tab1.write("The alternatives are on a 5-point agreement scale.")
    tab1.write("""" 1 - Disagree
                    2 - Partially Disagree
                    3 - Neutral
                    4 - Partially Agree
                    5 - Agree""")
    tab1.write("---------------------------------------------------")

    tab2.write("The following questions test the Neuroticism Trait")
    tab2.write("The alternatives are on a 5-point agreement scale.")
    tab2.write("""" 1 - Disagree
                    2 - Partially Disagree
                    3 - Neutral
                    4 - Partially Agree
                    5 - Agree""")
    tab2.write("---------------------------------------------------")

    tab3.write("The following questions test the Agreeableness Trait")
    tab3.write("The alternatives are on a 5-point agreement scale.")
    tab3.write("""" 1 - Disagree
                    2 - Partially Disagree
                    3 - Neutral
                    4 - Partially Agree
                    5 - Agree""")
    tab3.write("---------------------------------------------------")

    tab4.write("The following questions test the Conscientiousness Trait")
    tab4.write("The alternatives are on a 5-point agreement scale.")
    tab4.write("""" 1 - Disagree
                    2 - Partially Disagree
                    3 - Neutral
                    4 - Partially Agree
                    5 - Agree""")
    tab4.write("---------------------------------------------------")

    tab5.write("The following questions test the Openness to Experience Trait")
    tab5.write("The alternatives are on a 5-point agreement scale.")
    tab5.write("""" 1 - Disagree
                    2 - Partially Disagree
                    3 - Neutral
                    4 - Partially Agree
                    5 - Agree""")
    tab5.write("---------------------------------------------------")

    tab6.write("Ready to see your results?")



    with tab1:

        EXT1 = st.radio('I am the life of the party', [1, 2, 3, 4, 5])
        EXT2 = st.radio('I dont talk a lot', [1, 2, 3, 4, 5])
        EXT3 = st.radio('I feel comfortable around people', [1, 2, 3, 4, 5])
        EXT4 = st.radio('I keep in the background', [1, 2, 3, 4, 5])
        EXT5 = st.radio('I start conversations', [1, 2, 3, 4, 5])
        EXT6 = st.radio('I have little to say', [1, 2, 3, 4, 5])
        EXT7 = st.radio('I talk to a lot of different people at parties', [1, 2, 3, 4, 5])
        EXT8 = st.radio('I dont like to draw attention to myself', [1, 2, 3, 4, 5])
        EXT9 = st.radio('I dont mind being the center of attention', [1, 2, 3, 4, 5])
        EXT10 = st.radio('I am quiet around strangers', [1, 2, 3, 4, 5])

        ext = [EXT1,EXT2,EXT3,EXT4,EXT5,EXT6,EXT7,EXT8,EXT9,EXT10]

    with tab2:

        EST1 = st.radio('I get stressed out easily', [1, 2, 3, 4, 5])
        EST2 = st.radio('I am relaxed most of the time', [1, 2, 3, 4, 5])
        EST3 = st.radio('I worry about things', [1, 2, 3, 4, 5])
        EST4 = st.radio('I seldom feel blue', [1, 2, 3, 4, 5])
        EST5 = st.radio('I am easily disturbed', [1, 2, 3, 4, 5])
        EST6 = st.radio('I get upset easily', [1, 2, 3, 4, 5])
        EST7 = st.radio('I change my mood a lot', [1, 2, 3, 4, 5])
        EST8 = st.radio('I have frequent mood swings', [1, 2, 3, 4, 5])
        EST9 = st.radio('I get irritated easily', [1, 2, 3, 4, 5])
        EST10 = st.radio('I often feel blue', [1, 2, 3, 4, 5])

        est = [EST1,EST2,EST3,EST4,EST5,EST6,EST7,EST8,EST9,EST10]

    with tab3:

        AGR1 = st.radio('I feel little concern for others', [1, 2, 3, 4, 5])
        AGR2 = st.radio('I am interested in people', [1, 2, 3, 4, 5])
        AGR3 = st.radio('I insult people', [1, 2, 3, 4, 5])
        AGR4 = st.radio('I sympathize with others feelings', [1, 2, 3, 4, 5])
        AGR5 = st.radio('I am not interested in other peoples problems', [1, 2, 3, 4, 5])
        AGR6 = st.radio('I have a soft heart', [1, 2, 3, 4, 5])
        AGR7 = st.radio('I am not really interested in others', [1, 2, 3, 4, 5])
        AGR8 = st.radio('I take time out for others', [1, 2, 3, 4, 5])
        AGR9 = st.radio('I feel others emotions', [1, 2, 3, 4, 5])
        AGR10 = st.radio('I make people feel at ease', [1, 2, 3, 4, 5])

        agr = [AGR1,AGR2,AGR3,AGR4,AGR5,AGR6,AGR7,AGR8,AGR9,AGR10]

    with tab4:

        CSN1 = st.radio('I am always prepared', [1, 2, 3, 4, 5])
        CSN2 = st.radio('I leave my belongings around', [1, 2, 3, 4, 5])
        CSN3 = st.radio('I pay attention to details', [1, 2, 3, 4, 5])
        CSN4 = st.radio('I make a mess of things', [1, 2, 3, 4, 5])
        CSN5 = st.radio('I get chores done right away', [1, 2, 3, 4, 5])
        CSN6 = st.radio('I often forget to put things back in their proper place', [1, 2, 3, 4, 5])
        CSN7 = st.radio('I like order', [1, 2, 3, 4, 5])
        CSN8 = st.radio('I shirk my duties', [1, 2, 3, 4, 5])
        CSN9 = st.radio('I follow a schedule', [1, 2, 3, 4, 5])
        CSN10 = st.radio('I am exacting in my work', [1, 2, 3, 4, 5])

        csn = [CSN1,CSN2,CSN3,CSN4,CSN5,CSN6,CSN7,CSN8,CSN9,CSN10]

    with tab5:
        
        OPN1 = st.radio('I have a rich vocabulary', [1, 2, 3, 4, 5])
        OPN2 = st.radio('I have difficulty understanding abstract ideas', [1, 2, 3, 4, 5])
        OPN3 = st.radio('I have a vivid imagination', [1, 2, 3, 4, 5])
        OPN4 = st.radio('I am not interested in abstract ideas', [1, 2, 3, 4, 5])
        OPN5 = st.radio('I have excellent ideas', [1, 2, 3, 4, 5])
        OPN6 = st.radio('I do not have a good imagination', [1, 2, 3, 4, 5])
        OPN7 = st.radio('I am quick to understand things', [1, 2, 3, 4, 5])
        OPN8 = st.radio('I use difficult words', [1, 2, 3, 4, 5])
        OPN9 = st.radio('I spend time reflecting on things', [1, 2, 3, 4, 5])
        OPN10 = st.radio('I am full of ideas', [1, 2, 3, 4, 5])

        opn = [OPN1,OPN2,OPN3,OPN4,OPN5,OPN6,OPN7,OPN8,OPN9,OPN10]
     
    
    with tab6:
        if(st.button("Get Results")):
                ans = [(((sum(ext))/10)/5), (((sum(est))/10)/5), (((sum(agr))/10)/5),(((sum(csn))/10)/5), (((sum(opn))/10)/5)]
                st.write('Extraversion score:')
                st.write(ans[0])
                st.write('Neuroticism score:')
                st.write(ans[1])
                st.write('Agreeableness score:')
                st.write(ans[2])
                st.write('Conscientiousness score:')
                st.write(ans[3])
                st.write('Opennes to Experience score:')
                st.write(ans[4])

        st.markdown("Here's how to interpret the results:")
        st.write(""" 

1. Openness: People who score high on openness tend to be creative, imaginative, and curious. 
They enjoy exploring new ideas and are comfortable with ambiguity and uncertainty. 
They may be seen as unconventional or non-traditional in their approach to life.

2. Conscientiousness: Individuals who score high on conscientiousness are typically responsible, dependable, and organized. 
They tend to be achievement-oriented and value hard work and persistence. They may be seen as perfectionists or workaholics.

3. Extraversion: People who score high on extraversion tend to be outgoing, sociable, and energetic. T
hey enjoy being around others and tend to be talkative and assertive. 
They may be seen as charismatic or attention-seeking.

4. Agreeableness: Individuals who score high on agreeableness tend to be cooperative, empathetic, and compassionate. 
They prioritize getting along with others and tend to be accommodating and supportive. They may be seen as friendly or too eager to please.

5. Neuroticism: People who score high on neuroticism tend to experience negative emotions such as anxiety, sadness, and anger more frequently and intensely than others. 
They may be seen as moody, anxious, or overly sensitive.

It's important to keep in mind that these traits are not necessarily good or bad, and everyone has a unique combination of them.
 Understanding your own personality traits can help you better understand your strengths and weaknesses, as well as how you interact with others.""")

def locations():
    st.write("""
    Our website provides a preliminary assesment of the personalities. 
    To get a more detailed analysis, we recommend you to visit a professional specialising in psychiatrics.
    Select your city of choice.""")

    ct1, ct2, ct3, ct4, ct5, ct6, ct7, ct8, ct9 = st.tabs(["Mumbai", "Central Mumbai", "Thane", "Navi Mumbai", "Pune", "Delhi", "Chennai", "Patna", "Hyderabad"])
    
    mumbai = pd.read_csv('Mumbai.csv')
    cent_mumbai = pd.read_csv('Central Mumbai.csv')
    thane = pd.read_csv('Thane.csv')
    navi_mumbai = pd.read_csv('Navi Mumbai.csv')
    pune = pd.read_csv('Pune.csv')
    delhi = pd.read_csv('Delhi.csv')
    chennai = pd.read_csv('Chennai.csv')
    patna = pd.read_csv('Patna.csv')
    hyd = pd.read_csv('Hyderabad.csv')

    with ct1:
        st.map(mumbai['Name', 'lat', 'lon'])
        st.write(mumbai)
    
    with ct2:
        st.map(cent_mumbai['Name', 'lat', 'lon'])
        st.write(cent_mumbai)

    with ct3:
        st.map(thane['Name', 'lat', 'lon'])
        st.write(thane)

    with ct4:
        st.map(navi_mumbai['Name', 'lat', 'lon'])
        st.write(navi_mumbai)

    with ct5:
        st.map(pune['Name', 'lat', 'lon'])
        st.write(pune)

    with ct6:
        st.map(delhi['Name', 'lat', 'lon'])
        st.write(delhi)

    with ct7:
        st.map(chennai['Name', 'lat', 'lon'])
        st.write(chennai)

    with ct8:
        st.map(patna['Name', 'lat', 'lon'])
        st.write(patna)

    with ct9:
        st.map(hyd['Name', 'lat', 'lon'])
        st.write(hyd)
    


page_names_to_funcs = {
    "Home": main_page,
    "Give the test": test,
    "Find a Professional": locations,
}
selected_page = st.sidebar.selectbox("Explore the site", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
