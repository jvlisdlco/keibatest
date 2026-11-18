st.markdown("<div class='section-title'>💰 推奨買い目</div>", unsafe_allow_html=True)

axis_nums = axis_candidates["umaban"].tolist()
partner_nums = partners["umaban"].tolist()
overlap = set(axis_nums) & set(partner_nums)
unique_partners = [p for p in partner_nums if p not in axis_nums]

col_a, col_b = st.columns(2)

with col_a:
    if len(overlap) >= 2:
        box_nums = list(overlap)[:4]
        st.markdown(f"""
        <div class='bet-box'>
            <b>ワイド BOX</b><br>
            {' - '.join(map(str, box_nums))}（{len(box_nums)*(len(box_nums)-1)//2}点）
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='bet-box'>
            <b>ワイド</b><br>
            軸 {axis_nums[0]} → {unique_partners[:3]}
        </div>
        """, unsafe_allow_html=True)

with col_b:
    if len(overlap) >= 2:
        box_nums = list(overlap)[:4]
        st.markdown(f"""
        <div class='bet-box'>
            <b>馬連 BOX</b><br>
            {' - '.join(map(str, box_nums))}（{len(box_nums)*(len(box_nums)-1)//2}点）
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='bet-box'>
            <b>馬連</b><br>
            軸 {axis_nums[0]} → {unique_partners[:3]}
        </div>
        """, unsafe_allow_html=True)
