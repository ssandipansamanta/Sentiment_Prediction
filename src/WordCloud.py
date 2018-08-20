def WordCloud(Output_Path,FileName,data,type):
    # data = RawData[RawData.Sentiment == 1].reset_index(drop=True)
    # type='Positive'
    WC_words = ' '
    for Feedback in data.User_Feedback:
        Feedback = str(Feedback)
        tokens = Feedback.split()
        for words in tokens:
            WC_words = WC_words + words + ' '

    from wordcloud import WordCloud
    wordcloud = WordCloud(width=800, height=800,background_color='white',min_font_size=10).generate(WC_words)

    bp = PdfPages(Output_Path + FileName + '_' +type +'_WordCloud.pdf')
    fig, axes = plt.subplots(figsize=(8, 8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    bp.savefig(fig)
    bp.close()

    return None